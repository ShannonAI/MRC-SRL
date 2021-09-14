import sys
import json
from tqdm import tqdm

def process(input_path,output_path):
    data = [json.loads(d) for d in open(input_path).readlines()]
    all_sentences = []
    all_predicates = []
    all_args = []
    all_lemmas = []
    all_frameset_ids = []
    for d in tqdm(data):
        sentences = d['sentences']
        srl = d['srl']
        lemmas = d['lemma']
        frameset_ids = d['frameset_id']
        sent_offs = [len(sum(sentences[:i],[])) for i in range(len(sentences))]
        for i,s in enumerate(srl):
            t = []
            sent_off = sent_offs[i]
            for predicate,start,end,label in s:
                p1 = predicate-sent_off
                s1 = start-sent_off
                e1 = end - sent_off
                t.append([p1,s1,e1,label])
            pres = sorted(list(set([i[0] for i in t])))
            args = [[] for _ in pres]
            for ti in t:
                idx = pres.index(ti[0])
                args[idx].append(ti[1:])
            lemma = [lemmas[i][p] for p in pres]
            frameset_id = [frameset_ids[i][p] for p in pres]
            all_predicates.append(pres)
            all_args.append(args)
            all_sentences.append(sentences[i])
            all_lemmas.append(lemma)
            all_frameset_ids.append(frameset_id)
    all_data = []
    for sent,pre,arg,lemmas,frameset_ids in zip(all_sentences,all_predicates,all_args,all_lemmas,all_frameset_ids):
        all_data.append({'sentence':sent,'predicates':pre,'arguments':arg,'lemmas':lemmas,'frameset_ids':frameset_ids})
    with open(output_path,'w') as f:
        json.dump(all_data,f,sort_keys=True,indent=4)

if __name__=="__main__":
    process(sys.argv[1],sys.argv[2])