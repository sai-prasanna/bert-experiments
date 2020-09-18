import os
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import torch
import torch.utils.data
from ignite.engine import Engine, Events
from ignite.metrics import Loss, Accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import normalize
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Net(torch.nn.Module):
    def __init__(self, nb_classes):
        super().__init__()

        self.nb_classes = nb_classes

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(16, 32, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 16, 1),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(16, self.nb_classes, 1),
        )

    def forward(self, inputs):
        h = self.net(inputs)
        logits = h.squeeze(dim=-1).squeeze(dim=-1)
        return logits


def init_weights(modules):
    if isinstance(modules, torch.nn.Module):
        modules = modules.modules()

    for m in modules:
        if isinstance(m, torch.nn.Sequential):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.ModuleList):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
            torch.nn.init.xavier_normal_(m.weight.data)
            # m.bias.data.zero_()
            if m.bias is not None:
                m.bias.data.normal_(0, 0.01)

        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()


def pad_image(image, max_size):
    image = image[:max_size, :max_size]

    pad_h = (max_size - image.shape[0]) // 2
    pad_h = (pad_h, max_size - pad_h - image.shape[0])

    pad_w = (max_size - image.shape[1]) // 2
    pad_w = (pad_w, max_size - pad_w - image.shape[1])

    image = np.pad(image, (pad_h, pad_w), 'constant', constant_values=0)

    return image


def standartize_image_name(image_name):
    image_name = image_name.strip()

    if not '_' in image_name:
        # need to pad 
        min_len = 4
        nb_pad = min_len - len(image_name)
        image_name = ('0' * nb_pad) + image_name

    return image_name

def load_data(images_dir, labels_filename=None, only_labels=True, label2id=None, min_max_size=None, sample=None):
    # load labels
    labels_dict = {}
    if labels_filename is not None:
        with open(labels_filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                image_name, label = line.split("\t")
                labels_dict.update({image_name: label})

        if label2id is None:
            label2id = {l: i for i, l in enumerate(sorted(set(labels_dict.values())))}

    # load images
    images = []
    labels = []

    images_files = sorted(images_dir.iterdir())
    print(f'Images files: {len(images_files)}')

    if sample is not None:
        images_files = random.sample(images_files, min(sample, len(images_files)))
        print(f'Images files sampled: {len(images_files)}')

    for img_file in tqdm(images_files, desc='Reading'):
        img_name = img_file.stem

        if only_labels and img_name not in labels_dict:
            continue

        img = np.load(img_file)
        images.append(img)
        labels.append(label2id[labels_dict[img_name]] if img_name in labels_dict else -1)

    print(f'Images: {len(images)}, labels: {len(labels)}')

    # pad images
    if min_max_size is None:
        min_size = min(img.shape[0] for img in images)
        max_size = max(img.shape[0] for img in images)
        min_max_size = (min_size, max_size,)
    else:
        min_size, max_size = min_max_size
    images = [pad_image(img, max_size) for img in tqdm(images, desc='Padding')]
    print(f'Min size: {min_size}, max size: {max_size}')

    images = np.stack(images)[:, np.newaxis, :, :].astype(np.float32)
    labels = np.stack(labels).astype(np.long)
    print(f'Images data: {images.shape}, labels data: {labels.shape}')
    print(f'Images min: {images.min()}, images max: {images.max()}')
    print(f'Labels min: {labels.min()}, labels max: {labels.max()}')

    return images, labels, label2id, min_max_size


def plot_confusion_matrix(model, data_loader, device, label2id):
    y_true, y_pred = predict(model, data_loader, device)

    labels = sorted([l for l in label2id.keys()], key=lambda l: label2id[l])
    clf_report = classification_report(y_true, y_pred, target_names=labels)
    print(clf_report)


def predict(model, data_loader, device):
    y_true = []
    y_pred = []
    for batch in data_loader:
        inputs, targets = [x.to(device) for x in batch]
        outputs = model(inputs)

        y_true.append(targets.cpu().numpy())
        y_pred.append(outputs.argmax(-1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    print(f'Y true: {y_true.shape}, Y pred: {y_pred.shape}')

    return y_true, y_pred


def format_metrics_str(metrics):
    log_str = []

    for m_name, m_value in metrics.items():
        log_str.append(f'{m_name} {m_value:.3f}')

    log_str = ', '.join(log_str)
    return log_str


def save_model(filename, model, label2id, min_max_size):
    state_dict = {
        'model': model.state_dict(),
        'label2id': label2id,
        'min_max_size': min_max_size,
    }
    torch.save(state_dict, filename)


def load_model(filename):
    state_dict = torch.load(filename)

    label2id = state_dict['label2id']
    min_max_size = state_dict['min_max_size']

    model = Net(len(label2id))
    model.load_state_dict(state_dict['model'])

    return model, label2id, min_max_size


def main():
    set_seed(13)

    batch_size = 64
    nb_epochs = 200
    val_set_size = 0.2
    print_report = True

    data_dir = Path('../data/head_classification_data/normed/')
    train_dir = data_dir.joinpath('data/')
    labels_filename = data_dir.joinpath('attention_norm_annotated.tsv')
    model_filename = '../models/head_classifier/classify_normed_patterns.tar'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    images, labels, label2id, min_max_size = load_data(train_dir, labels_filename)

    if val_set_size > 0:
        images_train, images_val, labels_train, labels_val = train_test_split(
            images, labels, test_size=val_set_size, stratify=labels
        )
    else:
        images_train, labels_train = images, labels
        images_val, labels_val = None, None

    print(f'Train: {images_train.shape} {labels_train.shape}')
    if labels_val is not None:
        print(f'Val: {images_val.shape}, {labels_val.shape}')

    dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(images_train), torch.from_numpy(labels_train))
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    if labels_val is not None:
        dataset_val = torch.utils.data.TensorDataset(torch.from_numpy(images_val), torch.from_numpy(labels_val))
        data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    model = Net(len(label2id))
    model = model.to(device)
    init_weights(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    def update_function(engine, batch):
        model.train()
        optimizer.zero_grad()

        inputs, targets = [x.to(device) for x in batch]

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        return loss

    def inference_function(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = [x.to(device) for x in batch]

            outputs = model(inputs)

            return outputs, targets

    trainer = Engine(update_function)
    evaluator = Engine(inference_function)

    metrics = [
        ('loss', Loss(torch.nn.CrossEntropyLoss())),
        ('accuracy', Accuracy()),
    ]
    for name, metric in metrics:
        metric.attach(evaluator, name)

    best_val_acc = 0
    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        nonlocal best_val_acc

        evaluator.run(data_loader_train)
        metrics_train = format_metrics_str(evaluator.state.metrics)

        if labels_val is not None:
            evaluator.run(data_loader_val)
            metrics_val = format_metrics_str(evaluator.state.metrics)

            acc_val = evaluator.state.metrics['accuracy']
            if acc_val >= best_val_acc:
                save_model(model_filename, model, label2id, min_max_size)
                best_val_acc = acc_val
        else:
            metrics_val = {}

        print(f'Epoch: {engine.state.epoch} | Train: {metrics_train} | Val: {metrics_val}')

    trainer.run(data_loader_train, max_epochs=nb_epochs)

    if labels_val is None:
        save_model(model_filename, model, label2id, min_max_size)
    else:
        print(f'Best val accuracy: {best_val_acc}')

    if print_report:
        print(f'Train classification report')
        plot_confusion_matrix(model, data_loader_train, device, label2id)

        print(f'Val classification report')
        if labels_val is not None:
            plot_confusion_matrix(model, data_loader_val, device, label2id)


if __name__ == "__main__":
    main()
