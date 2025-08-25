import time
import torch, sys, os, tqdm, soundfile, time, pickle
import torch.nn as nn
import numpy as np
from tools import *
from loss import AAMsoftmax, OriginalSoftmax, RangeLoss
from model import ECAPA_TDNN, ECAPA_TDNN_Abla
import tqdm
import wfdb
import matplotlib.pyplot as plt
from tools import denoise, myOwnMelSprctrum, myOwnMFCC
torch.autograd.set_detect_anomaly(True)


class ECAPAModel_ECG(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        super(ECAPAModel_ECG, self).__init__()
        ## ECAPA-TDNN
        # self.ecg_encoder = ECAPA_TDNN(C=C).cuda()
        self.ecg_encoder = ECAPA_TDNN(C=C).cuda()
        ## Classifier
        self.ecg_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()
        # self.original_loss = OriginalSoftmax(n_class=n_class).cuda()

        # self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        # self.optim = torch.optim.Adam(self.ecg_encoder.parameters(), lr=lr, weight_decay=2e-5)
        self.optim = torch.optim.SGD(self.ecg_encoder.parameters(), lr=lr, weight_decay=2e-5)
        # print(self.ecg_encoder.parameters() == self.parameters())
        # exit()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
                sum(param.numel() for param in self.ecg_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        ## Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            ecg_embedding = self.ecg_encoder.forward(data.cuda(), aug=False)
            nloss, prec = self.ecg_loss.forward(ecg_embedding, labels)
            # print(nloss)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)

    def train_network_withEER(self, epoch, loader):
        self.train()
        ## Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data1, data_pos, data_neg, labels) in enumerate(loader, start=1):
            # for name, para in self.ecg_encoder.named_parameters():
            # 	print(name)
            # 	print(para.shape, para.requires_grad, para.grad)
            # exit()
            self.optim.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            ecg_embedding = self.ecg_encoder.forward(data1.cuda(), aug=False)
            pos_embedding = self.ecg_encoder.forward(data_pos.cuda(), aug=False)
            neg_embedding = self.ecg_encoder.forward(data_neg.cuda(), aug=False)
            nloss, prec = self.ecg_loss.forward(ecg_embedding, labels)
            # nloss, prec = self.original_loss.forward(ecg_embedding, labels)
            # print(torch.isnan(nloss).any())
            # if torch.isnan(nloss).any() == True:
            #     plt.plot(data1[0].reshape(4000).numpy())
            #     plt.show()
            # print('Input data:', torch.isnan(data1).any())
            # nloss.backward()
            trble_loss = nn.TripletMarginLoss()
            # print(ecg_embedding)
            alpha = 0.8
            triple = trble_loss(anchor=ecg_embedding, positive=pos_embedding, negative=neg_embedding)
            Loss_all = alpha * nloss + (1 - alpha) * triple
            # print(f'nloss {nloss}')
            # print(f'trble_loss {triple}')
            Loss_all.backward()
            # for name, para in self.ecg_encoder.named_parameters():
            # 	if name == 'bn6.bias':
            # 		print(para.shape, para.requires_grad, para.grad)
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += Loss_all.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)

    def train_network_withEER_range(self, epoch, loader):
        self.train()
        ## Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data1, data_pos, data_neg, labels) in enumerate(loader, start=1):
            self.optim.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            ecg_embedding = self.ecg_encoder.forward(data1.cuda(), aug=False)
            pos_embedding = self.ecg_encoder.forward(data_pos.cuda(), aug=False)
            neg_embedding = self.ecg_encoder.forward(data_neg.cuda(), aug=False)
            nloss, prec = self.ecg_loss.forward(ecg_embedding, labels)

            range_loss = RangeLoss()
            ran = range_loss(anchor=ecg_embedding, positive=pos_embedding, targets=labels)
            Loss_all = 0.5 * nloss + 0.5 * ran
            # exit()
            # print(f'nloss {nloss}')
            # print(f'trble_loss {triple}')
            Loss_all.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += Loss_all.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)

    def detect_nan(self, epoch, loader):
        self.train()
        ## Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        count = []
        num = 0
        for num, (data1, data_pos, data_neg, labels) in enumerate(loader, start=1):
            # for name, para in self.ecg_encoder.named_parameters():
            # 	print(name)
            # 	print(para.shape, para.requires_grad, para.grad)
            # exit()
            self.optim.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            ecg_embedding = self.ecg_encoder.forward(data1.cuda(), aug=False)
            pos_embedding = self.ecg_encoder.forward(data_pos.cuda(), aug=False)
            neg_embedding = self.ecg_encoder.forward(data_neg.cuda(), aug=False)
            nloss, prec = self.ecg_loss.forward(ecg_embedding, labels)
            print(torch.isnan(nloss).any())
            if torch.isnan(nloss).any() == True:
                count.append(num)
            # print('Input data:', torch.isnan(data1).any())
            # nloss.backward()
            trble_loss = nn.TripletMarginLoss()
            # print(ecg_embedding)
            triple = trble_loss(anchor=ecg_embedding, positive=pos_embedding, negative=neg_embedding)
            Loss_all = 0.1*nloss + 0.9*triple
            # print(f'nloss {nloss}')
            # print(f'trble_loss {triple}')
            Loss_all.backward()
            # for name, para in self.ecg_encoder.named_parameters():
            # 	if name == 'bn6.bias':
            # 		print(para.shape, para.requires_grad, para.grad)
            # print(name)
            # print(para.shape, para.requires_grad, para.grad)
            # exit()
            num += 2
            # torch.nn.utils.clip_grad_norm_(self.ecg_encoder.parameters(), max_norm=1.0)
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += Loss_all.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        np.save('nan,npy', count)
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list, length, preprocess):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list, encoding='utf-8').read().splitlines()
        for line in lines:
            files.append(line.split('|')[1])
            files.append(line.split('|')[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            filename = file.split(' ')[0]
            # print(filename)
            audio = wfdb.rdrecord(filename).p_signal
            pos = int(file.split(' ')[1])
            # Full utterance
            # print(pos)
            audio_1 = audio[(pos - 1) * length: pos * length, 0]
            # print(len(audio_1))
            if preprocess:
                audio_1 = denoise(audio_1)
            audio_2 = audio[pos * length: (pos + 1) * length, 0]
            if preprocess:
                audio_2 = denoise(audio_2)
            data_1 = torch.FloatTensor(numpy.stack([audio_1], axis=0)).cuda()
            data_2 = torch.FloatTensor(numpy.stack([audio_2], axis=0)).cuda()
            data_1 = data_1.unsqueeze(dim=0)
            data_2 = data_2.unsqueeze(dim=0)
            # data_1 = torch.FloatTensor(FBank(audio_1, fs=360)).unsqueeze(dim=0).cuda()
            # data_2 = torch.FloatTensor(FBank(audio_2, fs=360)).unsqueeze(dim=0).cuda()
            # data_1 = myOwnMFCC(torch.tensor(audio_1.copy()), sample_rate=1000).permute((1, 0)).unsqueeze(dim=0).float().cuda()
            # data_2 = myOwnMFCC(torch.tensor(audio_2.copy()), sample_rate=1000).permute((1, 0)).unsqueeze(dim=0).float().cuda()
            # print(data_1.shape, data_2.shape)
            # ecg embeddings
            with torch.no_grad():
                embedding_1 = self.ecg_encoder.forward(data_1, aug=False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.ecg_encoder.forward(data_2, aug=False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        # embeddings[file] = [embedding_1, embedding_1]
        scores, labels = [], []

        count = 0
        pdist = nn.PairwiseDistance(p=2)
        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split('|')[1]]
            embedding_21, embedding_22 = embeddings[line.split('|')[2]]
            # Compute the scores
            # print(embedding_11)
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            # score_1 = torch.mean(pdist(embedding_11, embedding_21))
            # score_2 = torch.mean(pdist(embedding_12, embedding_22))
            score = (score_1 + score_2) / 2
            # print(score)
            score = score.detach().cpu().numpy()
            # score = score_1.detach().cpu().numpy()
            # if np.isnan(score):
            # 	count += 1
            if np.isnan(score) != True:
                scores.append(score)
                labels.append(int(line.split('|')[0]))
        # Coumpute EER and minDCF
        print(len(scores), len(labels))
        # print(scores)
        # scores = np.array(scores)
        # print(np.isinf(scores).any(), np.isfinite(scores).all(), np.isnan(scores).any())
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        return EER, minDCF

    def test_network_acc(self, epoch, loader):
        self.eval()
        self.cuda()
        ## Update the learning rate based on the current epcoh
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        self.ecg_encoder.cpu()
        count = 0
        at = 0
        for num, (data, labels) in tqdm.tqdm(enumerate(loader, start=1)):
            count += 16
            st = time.time()
            labels = torch.LongTensor(labels).cuda()
            ecg_embedding = self.ecg_encoder.forward(data, aug=False).cuda()
            nloss, prec = self.ecg_loss.forward(ecg_embedding, labels)
            et = time.time()
            at = et-st
            # nloss, prec = self.original_loss.forward(ecg_embedding, labels)
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), top1 / index * len(labels)))
            sys.stderr.flush()

        print('Average time: ', 1000*at/count)
        sys.stdout.write("\n")
        return loss / num, top1 / index * len(labels)

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

    def feature_label(self, loader):
        embed = []
        label = []
        for num, (data, labels) in tqdm.tqdm(enumerate(loader, start=1)):
            self.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            ecg_embedding = self.ecg_encoder.forward(data.cuda(), aug=True)
            embed.append(ecg_embedding.detach().cpu().numpy())
            label.append(labels.cpu().numpy())
        np.save('embed.npy', np.concatenate(embed))
        np.save('embed_label.npy', np.concatenate(label))
        return np.concatenate(embed), np.concatenate(label)

