import numpy as np
import torch
import classification_model
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import feature_extractor as fe
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = ""
dev_name = ""
save_pre = "clear"

class fingerprint_dataset(Dataset):
    def __init__(self, f_name, trainning):
        raw_data = np.load(f_name)
        self.trainning = trainning
        self.data = raw_data["data"]
        self.label = raw_data["label"]
        if trainning:
            self.raw_signal = raw_data["raw_signal"]
            self.start_time = raw_data["signal_start_time"]
            self.phase_amplitude = raw_data["phase_amplitude"]
        if self.label.shape[0] != self.data.shape[0]:
            raise Exception("size does not match: data:{0}, label:{1}".format(self.data.shape[0], len(self.label)))
        self.label = torch.tensor(self.label, dtype=torch.float32)
        self.data = torch.unsqueeze(torch.tensor(self.data, dtype=torch.float32), 1).contiguous()
        self.types = torch.argmax(self.label, dim=1)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.trainning:
            noise = np.random.randn(2, self.raw_signal.shape[1]).astype(np.float32)
            noise_signal = self.raw_signal[index:index+1, :].copy()
            max_amp = np.max([np.abs(noise_signal.real), np.abs(noise_signal.imag)])
            amplitude = np.random.rand(2, 1) * max_amp / 3
            noise = amplitude * noise
            noise_signal.real = noise_signal.real + noise[0, :]
            noise_signal.imag = noise_signal.imag + noise[1, :]
            ramp_seg = fe.rampseg_from_rawseg(raw_signal_seg=noise_signal, signal_start_time_in_seg=self.start_time[index:index+1], phase_amplitude=self.phase_amplitude[index:index+1])
            # self.data[index, :],
            return torch.from_numpy(ramp_seg), self.label[index, :], self.types[index]
        else:
            return self.data[index, :], self.label[index, :], self.types[index]
    
    def get_type_num(self):
        return self.label.shape[1]

class fingerprint_others_dataset(Dataset):
    def __init__(self, f_name,  in_chan, include=None, exclude=None, train=None):
        raw_data = np.load(f_name)
        self.data = torch.tensor(raw_data["data"], dtype=torch.float32)
        self.origin_label = torch.tensor(raw_data["label"], dtype=torch.float32)
        if include is not None and exclude is None:
            exclude = []
            for i in range(self.origin_label.shape[1]):
                if i not in include:
                    exclude.append(i)
        elif exclude is not None and include is None:
            include = []
            for i in range(self.origin_label.shape[1]):
                if i not in exclude:
                    include.append(i)
        elif include is None and exclude is None:
            raise Exception("No include and exclude")
    
        self.label = torch.hstack([self.origin_label[:, include], torch.zeros([self.origin_label.shape[0], 1], dtype=torch.float32)])
        if len(exclude) == 1:
            self.label[:, -1] == self.origin_label[:, exclude[0]]
        elif len(exclude) > 1:
            self.label[:, -1] = torch.sum(self.origin_label[:, exclude], dim=1)
        if torch.max(self.label[:, -1]) > 1:
            raise Exception("exclude has larger than 1")
    
        # remmove the data not for trainning
        label_sum = torch.sum(self.label, dim=1)
        valid_idx = torch.where(label_sum==1)[0]
        self.data = self.data[valid_idx, :]
        self.label = self.label[valid_idx, :]
        self.origin_label = self.origin_label[valid_idx, :]
        self.types = torch.argmax(self.origin_label, dim=1)

        if train is not None:
            valid_idx = []
            # databuf = []
            # labelbuf = []
            # typebuf = []
            for i in range(self.origin_label.shape[1]):
                all_type_idx = torch.where(self.origin_label[:, i]==1)[0]
                train_num = int(all_type_idx.shape[0] * 0.8)
                test_num = all_type_idx.shape[0] - train_num
                if train:
                    valid_idx.append(all_type_idx[:train_num])
                    # databuf.append(self.data[all_type_idx[:train_num]])
                    # labelbuf.append(self.label[all_type_idx[:train_num]])
                    # typebuf.append(self.types[all_type_idx[:train_num]])
                else:
                    valid_idx.append(all_type_idx[train_num:train_num+test_num])
                    # databuf.append(self.data[all_type_idx[train_num:train_num+test_num]])
                    # labelbuf.append(self.label[all_type_idx[train_num:train_num+test_num]])
                    # typebuf.append(self.types[all_type_idx[train_num:train_num+test_num]])
            
            valid_idx = torch.concatenate(valid_idx)
            self.data = self.data[valid_idx]
            self.label = self.label[valid_idx]
            self.types = self.types[valid_idx]

        if len(self.label) != self.data.shape[0]:
            raise Exception("size does not match: data:{0}, label:{1}".format(self.data.shape[0], len(self.label)))
        # self.data = torch.unsqueeze(self.data, 1).contiguous()
        self.in_chan = in_chan
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, self.in_chan, :], self.label[index, :], self.types[index]
    
    def get_type_num(self):
        return self.label.shape[1]


class fingerprint_group_dataset(Dataset):
    def __init__(self, f_name,  in_chan, group_size, include=None, exclude=None):
        raw_data = np.load(f_name)
        self.data = torch.tensor(raw_data["data"], dtype=torch.float32)
        self.origin_label = torch.tensor(raw_data["label"], dtype=torch.float32)
        if include is not None and exclude is None:
            exclude = []
            for i in range(self.origin_label.shape[1]):
                if i not in include:
                    exclude.append(i)
        elif exclude is not None and include is None:
            include = []
            for i in range(self.origin_label.shape[1]):
                if i not in exclude:
                    include.append(i)
        elif include is None and exclude is None:
            raise Exception("No include and exclude")
    
        self.label = torch.hstack([self.origin_label[:, include], torch.zeros([self.origin_label.shape[0], 1], dtype=torch.float32)])
        if len(exclude) == 1:
            self.label[:, -1] == self.origin_label[:, exclude[0]]
        elif len(exclude) > 1:
            self.label[:, -1] = torch.sum(self.origin_label[:, exclude], dim=1)
        if torch.max(self.label[:, -1]) > 1:
            raise Exception("exclude has larger than 1")
    
        # remmove the data not for trainning
        label_sum = torch.sum(self.label, dim=1)
        valid_idx = torch.where(label_sum==1)[0]
        self.data = self.data[valid_idx, :]
        self.label = self.label[valid_idx, :]
        self.origin_label = self.origin_label[valid_idx, :]
        self.types = torch.argmax(self.origin_label, dim=1)
        self.groups = []
        self.group_label = []
        self.group_type = []
        for t in range(self.origin_label.shape[1]):
            target_idx = torch.where(self.types==t)[0]
            target_data = self.data[target_idx]
            group_num = target_data.shape[0] // group_size
            for i in range(group_num):
                self.groups.append(target_data[i*group_size:(i+1)*group_size])
                self.group_label.append(self.label[target_idx[0]])
                self.group_type.append(t)
        self.groups = torch.tensor(np.array([item.numpy() for item in self.groups], dtype=np.float32), dtype=torch.float32)
        self.group_label = torch.tensor(np.array([item.numpy() for item in self.group_label], dtype=np.float32), dtype=torch.float32)
        self.group_type = torch.tensor(self.group_type)
        if self.group_label.shape[0] != self.groups.shape[0]:
            raise Exception("size does not match: data:{0}, label:{1}".format(self.data.shape[0], len(self.label)))
        # self.data = torch.unsqueeze(self.data, 1).contiguous()
        self.in_chan = in_chan
    
    def __len__(self):
        return self.groups.shape[0]

    def __getitem__(self, index):
        return self.groups[index, :, self.in_chan, :], self.group_label[index, :], self.group_type[index]
    
    def get_type_num(self):
        return self.label.shape[1]

def model_others_eval(classifier, test_dataset, alien_ids, group_size, confi_lim=0.5):
    b_size = 64 // group_size
    test_dataloader = DataLoader(test_dataset, batch_size=b_size, shuffle=False, drop_last=False)
    correct_num = 0
    total_num = 0
    correct_alien_num = 0
    total_alien_num = 0
    correct_target_num = 0
    total_target_num = 0
    correct_other_num = 0
    total_other_num = 0 

    classifier.eval()
    with torch.no_grad():
        for idx, (data, label, origin_label) in enumerate(test_dataloader):
            if data.ndim == 4:
                data = data.reshape([group_size*data.shape[0], data.shape[2], data.shape[3]])
            data = data.to(device)
            # print(label)
            out, _ = classifier(data)
            out = F.softmax(out, dim=1).cpu()
            # print(torch.max(out, dim=1))
            out = out.reshape([-1, group_size, out.shape[1]])
            max_poss, max_idx = torch.max(out, dim=2)
            max_idx[torch.where(max_poss<confi_lim)] = label.shape[1] - 1  # belongs to "others"
            predict_idx = torch.zeros(label.shape[0], dtype=torch.int32)
            vote = torch.zeros(label.shape[1])
            vote_poss = torch.zeros(label.shape[1])
            for i in range(max_idx.shape[0]):
                vote[:] = 0
                vote_poss[:] = 0
                for j in range(group_size):
                    vote[max_idx[i, j]] += 1
                    vote_poss += out[i, j, :]
                max_vote, max_vote_idx = torch.max(vote, dim=0)
                # Existing multiple has same vote
                max_vote_idx = torch.where(vote==max_vote)[0]
                if len(max_vote_idx) != 1:
                    max_vote_poss = vote_poss[max_vote_idx]
                    max_vote_idx = max_vote_idx[torch.argmax(max_vote_poss)]
                predict_idx[i] = max_vote_idx

            true_max_idx = torch.argmax(label, dim=1)
            total_num += label.shape[0]
            target_idx = torch.where(true_max_idx!=(label.shape[1]-1))[0]
            target_predict_label = predict_idx[target_idx]
            target_true_label = true_max_idx[target_idx]
            total_target_num += target_idx.shape[0]
            correct_target_num += int(torch.sum(torch.eq(target_predict_label, target_true_label)))
            for target_id in alien_ids:
                a_idx = torch.where(origin_label==target_id)[0]
                correct_alien_num += int(torch.sum(torch.eq(predict_idx[a_idx], label.shape[1] - 1)))
                total_alien_num += a_idx.shape[0]
            correct_num += int(torch.sum(torch.eq(predict_idx, true_max_idx)))
    total_other_num = total_num - total_target_num - total_alien_num
    correct_other_num = correct_num - correct_target_num - correct_alien_num

    # some test case do not include the others type 
    # Set the total to 1
    if total_alien_num == 0:
        alien_acc = -1
    else:
        alien_acc = correct_alien_num / total_alien_num
    if total_other_num == 0:
        other_acc = -1 
    else:
        other_acc = correct_other_num / total_other_num
    if (total_alien_num + total_other_num) == 0:
        all_other_acc = -1
    else:
        all_other_acc = (correct_other_num+correct_alien_num) / (total_other_num+total_alien_num)

    return correct_target_num / total_target_num, all_other_acc, other_acc, alien_acc, correct_num / total_num

    
def model_train_main(classifier, train_dataset, group_size, include, aliens, in_chan, epoch=20, step_num=-1, confi_lim=0.6):
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False, num_workers=4)
    optimizer_fe = torch.optim.Adam(classifier.parameters(), 1e-4, [0.5, 0.999])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer_fe, step_size=100, gamma=0.9)
    loss_label = torch.nn.CrossEntropyLoss(reduction='mean')
    step = 0
    for e in range(epoch):
        for idx, (data, label, _) in enumerate(train_dataloader):
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
                print("===========================================================")
                print("epoch:{}, step:{}, loss:{}".format(e, step, loss.item()))
                # if step < 200:
                #     continue
                # , 20, 25, 30, 35
                for dis in [16]:
                    if dis == 16:
                        test_data = fingerprint_others_dataset(dataset_path.format(dis), in_chan=in_chan, include=include, train=False)
                        acc = model_others_eval(classifier, test_data, alien_ids=aliens, group_size=1, confi_lim=confi_lim)
                    else:
                        test_data = fingerprint_group_dataset(dataset_path.format(dis), in_chan=in_chan, group_size=group_size, include=include)
                        # test_data = fingerprint_others_dataset(dataset_path.format(dis), in_chan=in_chan, include=include)
                        acc = model_others_eval(classifier, test_data, alien_ids=aliens, group_size=group_size, confi_lim=confi_lim)
                    print(dis, acc)
                    torch.save(classifier.state_dict(), "./models/{}_{}_classifer_{}.pt".format(save_pre, dev_name, step))
                classifier.train()

            if step_num != -1 and step == step_num:
                return classifier
    return classifier

def model_train_each(f_name, epoch=20, step_num=-1):
    train_dataset = fingerprint_dataset(f_name, False)
    return model_train_main(train_dataset, epoch, step_num)
    

def model_train_others(f_name, in_chan, include, exclude, alien, group_size, feat_len, epoch=20, step_num=-1, confi_lim=0.6):
    train_dataset = fingerprint_others_dataset(f_name, in_chan, include=include, exclude=exclude, train=True)
    classifier = classification_model.FeatureExtractor(train_dataset.get_type_num(), len(in_chan), feat_len=feat_len).to(device)
    return model_train_main(classifier, train_dataset, group_size, include, alien, in_chan, epoch, step_num, confi_lim)


feat_lens = {"nrf": 600,
             "esp":600,
             "cc2650":800,
             "da":800}

def nrf_main(load=False, group_size=3):
    # near dc = 8e5
    global dev_name, dataset_path
    dataset_path = "./data/nn_data/temp_nrf_{}c.npz"
    dev_name = "nrf"

    chan = [1]
    include = [0, 1]
    exlcude = [2, 3, 4, 5, 10, 11]
    alien = [6, 7, 8, 9]
    confi_lim = 0.8
    if load:
        params = torch.load("./models/temp_good/nrf/temp_nrf_classifer_750.pt")
        classifier = classification_model.FeatureExtractor(len(include)+1, len(chan), feat_len=feat_lens[dev_name]).to(device)
        classifier.load_state_dict(params)
    else:
        classifier = model_train_others(dataset_path.format(16), chan, include, exlcude, alien, group_size=group_size, feat_len=feat_lens[dev_name], epoch=60, confi_lim=confi_lim)
        torch.save(classifier.state_dict(), "./models/{}_clear_classifer_final.pt".format(dev_name))
    
    for dis in [16, 20, 25, 30, 35]:
        test_data = fingerprint_group_dataset(dataset_path.format(dis), in_chan=chan, group_size=group_size, include=include)
        acc = model_others_eval(classifier, test_data, alien_ids=alien, group_size=group_size, confi_lim=confi_lim)
        print(dis, acc)


def cc2650_main(load=False, group_size=3):
    global dev_name, dataset_path
    dataset_path = "./data/nn_data/temp_cc2650_{}c.npz"
    dev_name = "cc2650"

    chan = [4]
    include = [2]
    exlcude = [0, 1, 3]
    alien = [4]
    
    confi_lim = 0.95
    if load:
        params = torch.load("./models/temp_good/cc2650/temp_cc2650_classifer_500.pt")
        classifier = classification_model.FeatureExtractor(2, len(chan), feat_lens[dev_name]).to(device)
        classifier.load_state_dict(params)
    else:
        classifier = model_train_others(dataset_path.format(16), chan, include, exlcude, alien, group_size=group_size, feat_len=feat_lens[dev_name], epoch=60, confi_lim=confi_lim)
        torch.save(classifier.state_dict(), "./models/cc2650_temp_classifer_final.pt")
    
    for dis in [16, 20, 25, 30, 35]:
        test_data = fingerprint_group_dataset(dataset_path.format(dis), in_chan=chan, group_size=group_size, include=include)
        acc = model_others_eval(classifier, test_data, alien_ids=alien, group_size=group_size, confi_lim=confi_lim)
        print(dis, acc)

def da_main(load=False, group_size=1):
    global dev_name, dataset_path
    dataset_path = "./data/nn_data/temp_da_{}c.npz"
    dev_name = "da"

    chan = [1]
    include = [1, 2]
    exlcude = [0, 3, 4, 5]
    alien = [6]
    confi_lim = 0.8
    if load:
        params = torch.load("./models/temp_good/da/temp_da_classifer_1350.pt")
        classifier = classification_model.FeatureExtractor(len(include)+1, len(chan), feat_len=feat_lens[dev_name]).to(device)
        classifier.load_state_dict(params)
    else:
        classifier = model_train_others(dataset_path.format(16), chan, include, exlcude, alien, group_size=group_size, feat_len=feat_lens[dev_name], epoch=70, confi_lim=confi_lim)
        torch.save(classifier.state_dict(), "./models/{}_temp_classifer_final.pt".format(dev_name))
    
    for dis in [16, 20, 25, 30, 35]:
        test_data = fingerprint_group_dataset(dataset_path.format(dis), in_chan=chan, group_size=group_size, include=include)
        acc = model_others_eval(classifier, test_data, alien_ids=alien, group_size=group_size, confi_lim=confi_lim)
        print(dis, acc)

# import utils

if __name__ == "__main__":
    cc2650_main(True, 1)
    # nrf_main(True, 1)
    # da_main(True, 1)
