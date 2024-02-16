import numpy as np
import torch
import classification_model
import config 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class fingerprint_dataset(Dataset):
    def __init__(self, f_name, train, chan, target_id=None):
        f = np.load(f_name)
        self.data = torch.tensor(f["data"], dtype=torch.float32)
        self.label = torch.tensor(f["label"], dtype=torch.float32)
        if len(self.label) != self.data.shape[0]:
            raise Exception("size does not match: data:{0}, label:{1}".format(self.data.shape[0], len(self.label)))
        
        if train != 3:
            valid_idx = []
            for i in range(self.label.shape[1]):
                if target_id is not None and i not in target_id:
                    continue
                all_types_idx = torch.where(self.label[:, i]==1)[0]
                train_num = int(len(all_types_idx) * 0.8)
                if train==1:
                    valid_idx.append(all_types_idx[:train_num])
                elif train==0:
                    valid_idx.append(all_types_idx[train_num:])
            valid_idx = torch.concatenate(valid_idx)
            self.data = self.data[valid_idx]
            self.label = self.label[valid_idx]
        self.chan = chan

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, self.chan, :], self.label[index, :]

    def type_num(self):
        return self.label.shape[1]


def model_eval(fe, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)
    poss = []
    types = []
    fe.eval()
    with torch.no_grad():
        correct_num = 0
        total_num = 0
        for idx, (data, label) in enumerate(test_dataloader):
            data = data.to(device)
            label = label.to(device)
            out, _ = fe(data)
            out = F.softmax(out, dim=1)
            max_idx = torch.argmax(out, dim=1)
            true_max_idx = torch.argmax(label, dim=1)
            total_num += label.shape[0]
            correct_num += int(torch.sum(torch.eq(max_idx, true_max_idx)))
            poss.append(out.cpu().numpy())
            t = torch.argmax(label, dim=1)
            types.append(t.cpu().numpy())
    return correct_num / total_num, np.vstack(poss), np.concatenate(types)

def model_train_main(classifier, train_dataset, chan, pre_testdataset=None, epoch=20, step_num=-1):
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False, num_workers=4)
    optimizer_fe = torch.optim.Adam(classifier.parameters(), 1e-4, [0.5, 0.999])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer_fe, step_size=100, gamma=0.9)
    loss_label = torch.nn.CrossEntropyLoss(reduction='mean')
    step = 0
    for e in range(epoch):
        for idx, (data, label) in enumerate(train_dataloader):
            data = data.to(device)
            label = label.to(device)
            predicted_label, _ = classifier(data)
            loss = loss_label(predicted_label, label)
            classifier.zero_grad()
            loss.backward()
            optimizer_fe.step()
            step += 1
            # scheduler.step()
            if step % 50 == 0:
                if step < 200:
                    continue
                if pre_testdataset is not None:
                    acc, _, _ = model_eval(classifier, pre_testdataset)
                    print("===========================================================")
                    print("epoch:{}, step:{}, loss:{}, acc:{}".format(e, step, loss.item(), acc))
                    torch.save(classifier.state_dict(), "./models/class_dunique_{}_classifer_{}.pt".format(dev_name, step))
                else:
                    print("===========================================================")
                    print("epoch:{}, step:{}, loss:{}".format(e, step, loss.item()))
                    for temp in [16, 20, 25, 30, 35]:
                        if temp == 16:
                            test_dataset = fingerprint_dataset(dataset_path.format(temp), 0, chan)
                        else:
                            test_dataset = fingerprint_dataset(dataset_path.format(temp), 3, chan)
                        acc, _, _, eacc = model_eval(classifier, test_dataset)
                        print(temp, acc, eacc)
                    torch.save(classifier.state_dict(), "./models/unique_{}_classifer_{}.pt".format(dev_name, step))

                classifier.train()
            if step_num != -1 and step == step_num:
                return classifier
    for temp in [16, 20, 25, 30, 35]:
        test_dataset = fingerprint_dataset(dataset_path.format(temp), 3, chan)
        acc, _, _ = model_eval(classifier, test_dataset)
        print(temp, acc)
        torch.save(classifier.state_dict(), "./models/unique_{}_classifer_{}.pt".format(dev_name, step))
    return classifier

dev_name = "nrf"
# dataset_path = "/data/blueprint/data/nn_data/unique_nrf.npz"
# dataset_path = "/data/blueprint/data/nn_data/da_1m.npz"
# dataset_path = "/data/blueprint/data/nn_data/track_nrf_1.npz"
dataset_path = "/data/blueprint/data/nn_data/nrf_1m.npz"

chan = [1]
feat_len = 600
def train():
    train_dataset = fingerprint_dataset(dataset_path, 1, chan)
    test_dataset = fingerprint_dataset(dataset_path, 0, chan)
    # all_dataset = fingerprint_dataset(dataset_path, 3, chan)

    classifier = classification_model.FeatureExtractor(train_dataset.type_num(), len(chan), feat_len).to(device)
    classifier = model_train_main(classifier, train_dataset, chan, pre_testdataset=test_dataset, epoch=80, step_num=-1)

def nn_matching_rate(model_path):
    # both nn and cfo consider all dataset to show the performance of the learned device profile 
    all_dataset = fingerprint_dataset(dataset_path, 3, chan)
    print(len(all_dataset))
    classifer = classification_model.FeatureExtractor(all_dataset.type_num(), len(chan), feat_len).to(device)
    params = torch.load(model_path)
    classifer.load_state_dict(params)
    acc, poss, type = model_eval(classifer, all_dataset)
    matching_rates = []
    avg = []
    for i in range(all_dataset.type_num()):
        type_idx = np.where(type==i)[0]
        prd_poss = np.sum(poss[type_idx, :], axis=0) / len(type_idx)
        matching_rates.append(prd_poss)
        avg.append(prd_poss[i])
    print(np.mean(avg))
    return np.array(matching_rates)
# train()
# 1800
# 2000

#1950


# matching_rates = nn_matching_rate("./models/class_dunique_nrf_classifer_1000.pt")
# np.savetxt("./output/dis_nrf_nn_matching_rates.csv", matching_rates, delimiter=",")
# matching_rates = nn_matching_rate("./models/class_unique_nrf_classifer_1850.pt")
# np.savetxt("./output/track_nrf_nn_matching_rates.csv", matching_rates, delimiter=",")


feat_lens = {"nrf": 600,
             "esp":600,
             "cc2650":800,
             "da":800}
target_ids = {"nrf": [0, 1],
              "cc2650": [2],
              "da": [1, 2]}

def nrf_main(load=False):
    # near dc = 8e5
    global dev_name, dataset_path
    dev_name = "nrf"
    dataset_path = "/data/blueprint/data/nn_data/temp_{}_{}c.npz".format(dev_name, "{}")
    feat_len = feat_lens[dev_name]
    chan = [1]

    if load:
        params = torch.load("./models/each_fix/nrf/unique_nrf_classifer_700.pt")
        classifier = classification_model.FeatureExtractor(12, len(chan), feat_len=feat_lens[dev_name]).to(device)
        test_dataset = fingerprint_dataset(dataset_path.format(16), 0, chan)
        classifier.load_state_dict(params)
        acc, _, _ = model_eval(classifier, test_dataset)
        print(acc)
        # for temp in [16, 20, 25, 30, 35]:
        #     if temp == 16:
        #         test_dataset = fingerprint_dataset(dataset_path.format(temp), 0, chan, target_ids[dev_name])
        #     else:
        #         test_dataset = fingerprint_dataset(dataset_path.format(temp), 3, chan, target_ids[dev_name])
        #     acc, _, _ = model_eval(classifier, test_dataset)
        #     print(temp, acc)
    else:
        train_dataset = fingerprint_dataset(dataset_path.format(16), 1, chan)
        # test_dataset = fingerprint_dataset(dataset_path, 0, chan)
        classifier = classification_model.FeatureExtractor(train_dataset.type_num(), len(chan), feat_len).to(device)
        classifier = model_train_main(classifier, train_dataset, chan, epoch=30, step_num=-1)


def cc2650_main(load=False):
    # near dc = 8e5
    global dev_name, dataset_path
    dev_name = "cc2650"
    dataset_path = "/data/blueprint/data/nn_data/temp_{}_{}c.npz".format(dev_name, "{}")
    feat_len = feat_lens[dev_name]
    chan = [4]

    if load:
        params = torch.load("./models/each_fix/cc2650/unique_cc2650_classifer_1550.pt")
        classifier = classification_model.FeatureExtractor(5, len(chan), feat_len=feat_lens[dev_name]).to(device)
        classifier.load_state_dict(params)
        test_dataset = fingerprint_dataset(dataset_path.format(16), 0, chan)
        classifier.load_state_dict(params)
        acc, _, _ = model_eval(classifier, test_dataset)
        print(acc)

        # for temp in [16, 20, 25, 30, 35]:
        #     if temp == 16:
        #         test_dataset = fingerprint_dataset(dataset_path.format(temp), 0, chan, target_ids[dev_name])
        #     else:
        #         test_dataset = fingerprint_dataset(dataset_path.format(temp), 3, chan, target_ids[dev_name])
        #     acc, _, _ = model_eval(classifier, test_dataset)
        #     print(temp, acc)
    else:
        train_dataset = fingerprint_dataset(dataset_path.format(16), 1, chan)
        # test_dataset = fingerprint_dataset(dataset_path, 0, chan)
        classifier = classification_model.FeatureExtractor(train_dataset.type_num(), len(chan), feat_len).to(device)
        classifier = model_train_main(classifier, train_dataset, chan, epoch=50, step_num=-1)

def da_main(load=False):
    # near dc = 8e5
    global dev_name, dataset_path
    dev_name = "da"
    dataset_path = "/data/blueprint/data/nn_data/temp_{}_{}c.npz".format(dev_name, "{}")
    feat_len = feat_lens[dev_name]
    chan = [1]

    if load:
        params = torch.load("./models/each_fix/da/unique_da_classifer_900.pt")
        classifier = classification_model.FeatureExtractor(7, len(chan), feat_len=feat_lens[dev_name]).to(device)
        classifier.load_state_dict(params)
        test_dataset = fingerprint_dataset(dataset_path.format(16), 0, chan)
        classifier.load_state_dict(params)
        acc, _, _ = model_eval(classifier, test_dataset)
        print(acc)

        # for temp in [16, 20, 25, 30, 35]:
        #     if temp == 16:
        #         test_dataset = fingerprint_dataset(dataset_path.format(temp), 0, chan, target_ids[dev_name])
        #     else:
        #         test_dataset = fingerprint_dataset(dataset_path.format(temp), 3, chan, target_ids[dev_name])
        #     acc, _, _ = model_eval(classifier, test_dataset)
    else:
        train_dataset = fingerprint_dataset(dataset_path.format(16), 1, chan)
        # test_dataset = fingerprint_dataset(dataset_path, 0, chan)
        classifier = classification_model.FeatureExtractor(train_dataset.type_num(), len(chan), feat_len).to(device)
        classifier = model_train_main(classifier, train_dataset, chan, epoch=50, step_num=-1)

# cc2650_main(True)
nrf_main(True)
# da_main(True)