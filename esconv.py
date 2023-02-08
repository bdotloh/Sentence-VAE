import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text 
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

from torch.utils.data import Dataset



class ESCONV(Dataset):
    def __init__(self, data_dir='data/ESConv.json', load_split='train',  max_sequence_length=50):
        super().__init__()

        self.max_sequence_length=max_sequence_length
        self.es_conv=pd.read_json('data/ESConv.json')
       
        # i am only interested in supporter utterances for now
        self.es_conv['supporter_utterances']=self.es_conv.dialog.apply(lambda x : self._retrieve_utterances(x)['supporter'])
        supporter_utts=[utt for utts in self.es_conv.supporter_utterances.values.tolist() for utt in utts]

        train,test=train_test_split(supporter_utts,test_size=.15, random_state=42)
        train,valid=train_test_split(supporter_utts, test_size=.15, random_state=42)

        # print("Train:", len(train))
        # print("Test:", len(test))
        # print("Validation:", len(valid))

        # fit cv on train
        self.cv = sklearn.feature_extraction.text.CountVectorizer(lowercase=True)
        self.cv.fit(train)

        self.splits={
            'train':self._preprocess_and_tokenise_data(train),
            'valid': self._preprocess_and_tokenise_data(valid),
            'test':self._preprocess_and_tokenise_data(test)
            }


        #initialise vocab
        self._create_vocab()

        self.data=self._prepare_data(split=load_split)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }

    def _preprocess_and_tokenise_data(self,data):
        sk_word_tokenize = self.cv.build_tokenizer()
        sk_preprocesser = self.cv.build_preprocessor()
        tokenize = lambda doc: sk_word_tokenize(sk_preprocesser(doc))
        data_tokenised=[tokenize(dat) for dat in data]
        
        return data_tokenised

    def _retrieve_utterances(self, dialog):
        """
        takes a dialog, returns a speaker:[utt list] dict 
        """

        prev_speaker=None
        curr_speaker=None
    
        all_utterances=dict(zip(['seeker','supporter'],[[] for _ in range(2)]))

        for i,item in enumerate(dialog):
            prev_speaker=curr_speaker

            curr_speaker=item['speaker'].strip()
            curr_utt=item['content'].strip()
            
            if curr_speaker==prev_speaker:  # concat curr utterance to previous utterance"
                all_utterances[curr_speaker][-1]=f"{all_utterances[curr_speaker][-1]}. {curr_utt}"

            else:
                all_utterances[curr_speaker].append(curr_utt)
        
        #assert(len(all_utterances['seeker'])==len(all_utterances['supporter']))

        return all_utterances


    def _prepare_data(self,split='train'):
        #   NOTE: this is a baseline method that encodes tokens by labels. 
        #   This should probably be changed to some sort of embedding
        data=self.splits[split]
        
        data_processed = defaultdict(dict)
        inputs=[['<sos>']+tokens for tokens in data]
        inputs=[tokens[:self.max_sequence_length] for tokens in inputs]

        targets=[tokens[:self.max_sequence_length-1] for tokens in data]
        targets=[tokens+['<eos>'] for tokens in targets]

        lengths=[len(tokens) for tokens in inputs]

        [inputs[i].extend(['<pad>'] * (self.max_sequence_length-lengths[i])) for i in range(len(inputs))]
        [targets[i].extend(['<pad>'] * (self.max_sequence_length-lengths[i])) for i in range(len(targets))]

        inputs=[[self.w2i.get(w,self.w2i['<unk>']) for w in tokens] for tokens in inputs]
        targets=[[self.w2i.get(w,self.w2i['<unk>']) for w in tokens] for tokens in targets]

        for i in range(len(data)):
            data_processed[i]['input']=inputs[i]
            data_processed[i]['target']=targets[i]
            data_processed[i]['length']=lengths[i]
        
        return data_processed
    
    def _create_vocab(self):
        # create vocab from trainset
        self.vocab=dict(sorted(self.cv.vocabulary_.items(),key=lambda x: x[1]))
        # +4 to all indexs to make way for special tokens
        self.vocab={k:v+4 for k,v in self.vocab.items()}

        # add special tokens
        self.special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for i,token in enumerate(self.special_tokens):
            self.vocab[token]=i

        self.w2i=dict(sorted(self.vocab.items(),key=lambda x: x[1]))
        self.i2w={i:w for w,i in self.w2i.items()}

        # store w2i and i2w in dict
        self.vocab=dict(w2i=self.w2i, i2w=self.i2w)
    
    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='data/ESConv.json')
    parser.add_argument('--split',type=str,default='train')
    parser.add_argument('--max_seq_len',type=int,default=50)

    args = parser.parse_args()
    
    dataset=ESCONV(
        data_dir=args.data_dir, 
        load_split=args.split, 
        max_sequence_length=args.max_seq_len
        )

    print(f"number of datapoints in {args.split}: {len(dataset)}")
    print(f"vocab size: {dataset.vocab_size}")

