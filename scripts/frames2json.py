import os
import json
import argparse
from xml.dom.minidom import parse

def process_frames1(propbank_dir,nombank_dir='',pos=False):
    '''pos is used for CoNLL2012'''
    data = {}
    frames_dirs = [propbank_dir,nombank_dir] if nombank_dir!='' else [propbank_dir]
    for frames_dir in frames_dirs:
        files = sorted([i for i in os.listdir(frames_dir) if i.endswith('.xml')])
        for file in files:
            if pos:
                lemma0 = file.split('.')[0][:-2]
            else:
                lemma0 = file.split('.')[0]
            path = os.path.join(frames_dir,file)
            dom = parse(path)
            dom_data = dom.documentElement
            predicates = dom_data.getElementsByTagName('predicate')
            for predicate in predicates:
                lemma1 = predicate.getAttribute('lemma')
                rolesets = predicate.getElementsByTagName("roleset")
                for roleset in rolesets:
                    name = roleset.getAttribute('name')
                    _id =  roleset.getAttribute("id")
                    _id1 = lemma0+'.'+roleset.getAttribute("id").split('.')[-1]
                    lemma2,i =_id.split('.')
                    if _id1  not in data:
                        data[_id1] = {'meta':[],'args':{}}
                    # there are various forms of lemma in the frame files
                    data[_id1]['meta'].append((lemma0,lemma1,lemma2,i,name))
                    roles = roleset.getElementsByTagName('roles')
                    assert len(roles)==1
                    roles = roles[0].getElementsByTagName('role')
                    for role in roles:
                        n = role.getAttribute('n').upper()
                        if n in ['0','1','2','3','4','5','A']:
                            descr = role.getAttribute('descr')
                            if n not in data[_id1]['args']:
                                data[_id1]['args'][n]=[]
                            data[_id1]['args'][n].append(descr)
    return data


def process_frames2(data):
    data1 = {}
    for k,v in data.items():
        data1[k]={}
        meta = v['meta']
        args = v['args']
        meta1 = []
        if not len(meta)<=2:
            print('meta',meta)
        for m in meta:
            lemma = m[0] if '_' not in m[1] else m[1].replace('_',' ')
            meta1.append((lemma,m[-1]))
        if len(meta1)==1:
            meta2 = f'{meta1[0][0]}, {meta1[0][1]}'
        else:
            #mostly used for merging PropBank and NomBank
            m0,m1 = meta1[0][1].lower(),meta1[1][1].lower()
            if m0 in m1:
                meta2 = f'{meta1[1][0]}, {meta1[1][1]}'
            elif m1 in m0:
                meta2 = f'{meta1[0][0]}, {meta1[0][1]}'
            else:
                meta2 = f'{meta1[0][0]}, {meta1[0][1]}; {meta1[1][0]}, {meta1[1][1]}'
        args1 = {}
        for n,desc in args.items():
            desc = list(set(desc))
            if not len(desc)<=2:
                print("desc:",desc)
            if len(desc)==1:
                args1[n]=desc[0]
            else:
                d0,d1=desc[0].lower(),desc[1].lower()
                if d0 in d1:
                    args1[n]=d1
                elif d1 in d0:
                    args1[n]=d0
                else:
                    args1[n] = f'{d0} or {d1}'
        data1[k]['name']=meta2
        data1[k]['args']=args1
    return data1

if __name__=="__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('--propbank_dir')
    parser.add_argument('--nombank_dir',default='')
    parser.add_argument('--output_path')
    parser.add_argument("--pos",action='store_true')
    args = parser.parse_args()
    data = process_frames1(args.propbank_dir,args.nombank_dir,args.pos)
    data = process_frames2(data)
    with open(args.output_path,'w') as f:
        json.dump(data,f,sort_keys=True,indent=4)