import torch
import argparse
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint
from torch.optim import Adam
from tqdm import tqdm
from os.path import join
from time import time
from tensorboardX import SummaryWriter

from zoom import Configure, ZoomEncoder, ZoomAgent
from utils import set_global_logging_level, ctext, get_value, convert_label, compute_performance, check_gpu_device
from utils import setup_seed, clog

from data import TOFRData, DataLoader, collate


def load_data(root_path, max_length=-1):
    init_time = time()
    paths = [join(root_path, "{}.pkl".format(data_prex)) for data_prex in ['train', 'dev', 'test']]
    datasets = [TOFRData(data_path=path, max_length=ml) for path, ml in zip(paths, [max_length, -1, -1])]
    end_time = time()
    data = dict()
    for dataset, name in zip(datasets, ['train', 'dev', 'test']):
        data[name] = dataset
    print(ctext("Data Loaded ({} s)".format(end_time - init_time), 'green'))
    return data


def zoom_train(config, data, encoder, agent, encoder_optim, agent_optim, stage):
    encoder.train()
    agent.train()

    loss = 0.
    wlar = 0.
    counter = 0
    for batch in tqdm(data, desc='{} stage'.format(stage)):
        encoder.zero_grad()
        agent.zero_grad()
        words, sentences, paragraphs = encoder.forward(batch)
        batch_target = check_gpu_device(torch.tensor(0.))
        for sample_id in range(config.document_batch_size):
            supervised_loss, reward, log_sum_pie, word_action_ratio, _, _ = agent.sample_forward(
                content=batch['content'][sample_id],
                label=batch['zoom'][sample_id],
                memory=[words[sample_id], sentences[sample_id], paragraphs[sample_id]],
                sentence_boundaries=batch['sentence_boundaries'][sample_id],
                paragraph_boundaries=batch['paragraph_boundaries'][sample_id],
                mode='train',
                show_result=False,
                show_action_level=False
            )
            target = supervised_loss + config.rl_lambda * log_sum_pie * reward
            batch_target += target
            loss += get_value(supervised_loss)
            wlar += word_action_ratio
            counter += 1
        batch_target.backward()
        encoder_optim.step()
        agent_optim.step()
        batch_target.detach()
    info = {
        "loss": loss/counter,
        "wlar": wlar/counter
    }
    return info


def zoom_eval(config, data, encoder, agent, encoder_optim, agent_optim, stage):
    encoder.eval()
    agent.eval()

    loss = 0.
    wlar = 0.
    ground_labels = []
    pred_labels = []
    counter = 0
    with torch.no_grad():
        for batch in tqdm(data, desc='{} stage'.format(stage)):
            words, sentences, paragraphs = encoder.forward(batch)
            for sample_id in range(config.document_batch_size):
                supervised_loss, reward, log_sum_pie, word_action_ratio, pred_label, ground_label = agent.sample_forward(
                    content=batch['content'][sample_id],
                    label=batch['zoom'][sample_id],
                    memory=[words[sample_id], sentences[sample_id], paragraphs[sample_id]],
                    sentence_boundaries=batch['sentence_boundaries'][sample_id],
                    paragraph_boundaries=batch['paragraph_boundaries'][sample_id],
                    mode='eval',
                    show_result=False,
                    show_action_level=False
                )
                loss += get_value(supervised_loss)
                wlar += word_action_ratio
                ground_labels.append(convert_label(ground_label))
                pred_labels.append(convert_label(pred_label))
                counter += 1

    info = compute_performance(ground_labels, pred_labels, loss/counter, wlar/counter)
    return info


def process(config):
    set_global_logging_level()
    config.inform()
    setup_seed(config.seed)
    logger = SummaryWriter(logdir='./tensorboard_logs/{}'.format(config.tag))
    logger.add_text(tag='configure',
                    text_string=config.mark_down(),
                    global_step=0)
    torch.cuda.set_device(config.gpu_id)
    dataset = load_data(root_path=config.data, max_length=config.max_length)
    encoder = ZoomEncoder(config)
    agent = ZoomAgent(config)

    encoder_bert_params = list(map(id, encoder.bert_encoder.parameters()))
    encoder_other_params = filter(lambda param: id(param) not in encoder_bert_params,
                                  encoder.parameters())
    encoder_optim = Adam([
        {"params": encoder.bert_encoder.parameters(),
         "lr": config.bert_lr},
        {"params": encoder_other_params,
         "lr": config.learning_rate}
    ]
    )
    agent_optim = Adam(params=agent.parameters(),
                       lr=config.learning_rate,
                       weight_decay=config.weight_decay)

    function_dict = {
        'train': zoom_train,
        'dev': zoom_eval,
        'test': zoom_eval
    }
    stages = ['train', 'dev', 'test']
    colors = ['blue', 'green', 'yellow']
    previous_best = 0
    patience_counter = 0
    for epoch in range(config.max_epoch):
        print(ctext('===============epoch {} ==============='.format(epoch), 'red'))
        for stage, color in zip(stages, colors):
            data = DataLoader(dataset=dataset[stage],
                              batch_size=config.document_batch_size,
                              collate_fn=collate,
                              shuffle=True)
            info = function_dict[stage](
                config=config,
                data=data,
                encoder=encoder,
                agent=agent,
                encoder_optim=encoder_optim,
                agent_optim=agent_optim,
                stage=stage
            )
            for key in info.keys():
                logger.add_scalar(tag='{}/{}'.format(stage, key), scalar_value=info[key], global_step=epoch)
                clog(key='{}/{}'.format(stage, key).ljust(25),
                     value='{}'.format(info[key])[:9],
                     color=color)
                if stage == 'dev':
                    f1 = info['macro_f1']
                    if f1 > previous_best:
                        previous_best = f1
                        patience_counter = 0
                        encoder.save('best')
                        agent.save('best')
                    else:
                        patience_counter += 1

            if patience_counter >= config.early_stop_delay_margin:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configuration')
    parser.add_argument('-config',
                        type=str,
                        default='zoombert')
    args = parser.parse_args()
    config = Configure('./configs/{}.json'.format(args.config))
    process(config)

