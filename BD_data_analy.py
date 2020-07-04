import json
import torch

path_train = './data/train.json'
path_dev = './data/dev.json'
path_list = [path_train, path_dev]

events = []
for path in path_list:
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            sentence = data['text']
            words = [word for word in sentence]
            if len(words) > 500:
                print("===")

            for event_mention in data['event_list']:
                if event_mention['event_type'] not in events:
                    events.append(event_mention['event_type'])
            
triggers = {}
for trigger in events:
    triggers[trigger.split('-')[-1]] = trigger
    
file = './data/pred.json'
result = []
with open(file, mode='r', encoding='utf-8') as f:
    name = json.load(f)
    for data in name:
        item = {}
        item['id'] = data['id']
        item["event_list"] = []
        events = eval(data['event_list'])
       
        text = data['words']
        for trigger in events:
           
            single_event = {}
            st, ed, type = trigger
            single_event['event_type'] = triggers[type]
            single_event['arguments'] = []
            for argument in events[trigger]:
                single_argument = {}
                ast, aed, atype = argument
                single_argument['role'] = atype
                single_argument['argument'] = text[ast:aed]
                single_event['arguments'].append(single_argument)
            item["event_list"].append(single_event)
        result.append(item)

# solution 1 
# jsObj = json.dumps(result)   
with open('./data/pred_new.json', "w") as f:
    for item in result:
        jsObj = json.dumps(item, ensure_ascii=False, sort_keys=False)
        f.write('{}\n'.format(jsObj))  
f.close()
# for item in result:
#     with open('./data/pred_new.json', 'w') as f:
#         json.dump(result, f, indent=2, ensure_ascii=False,sort_keys=False)
# with open('./data/pred_new.json', 'w') as f:
#     json.dump(result, f, indent=2, ensure_ascii=False, sort_keys=False)
# file1 = './output_1/44_test'
# file2 = './data/test1.json'


# result = []
# with open(file1, mode='r', encoding='utf-8') as f:
#     events = []
#     for line in f.readlines():
#         if 'arguments' in line:
#             ev = line.split('#')[-1].strip()
#             events.append(ev)
# print(len(events))

# with open(file2, mode='r', encoding='utf-8') as f:
#     i = 0
#     for line in f.readlines():
#         item = {}
#         data = json.loads(line)
#         item['words'] = data['text']
#         item['id'] = data['id']
#         item['event_list'] = events[i]
#         result.append(item)
#         i = i + 1
        
# with open('./data/pred.json', 'w') as f:
#     json.dump(result, f, indent=2, ensure_ascii=False)
    
# from pytorch_pretrained_bert import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese-vocab.txt', do_lower_case=False)
# text = '一云南网红直播自己母亲葬礼，结果直播中突发意外，因为暴雨，雨棚倒塌，最终导致18人重伤2人轻伤，而主播在支付了部分医药费后，却玩起了失踪，甚至一度继续直播他人受伤，引发网友愤怒\n不少网友都不理解，直播母亲葬礼这个行为'
# text = text.replace('\n', '-')
# print(text)
# for t in text:
#     tokens = tokenizer.tokenize(t)
#     print(tokens)
#     tokens_xx = tokenizer.convert_tokens_to_ids(tokens)
#     print(tokens_xx)


# path_train = './data/train.json'
# path_dev = './data/dev.json'
# path_test = './data/test1.json'
# path_sample = './data/sample.json'
# path_list = [path_train, path_dev]
#
#
# events = []
# roles = []
# L = 0
# for path in path_list:
#     with open(path, mode='r', encoding='utf-8') as f:
#         for line in f.readlines():
#             data = json.loads(line)
#             sentence = data['text']
#             words = [word for word in sentence]
#             if len(words) > 500:
#                 print("===")
#
#             Entity = {}
#             for event_mention in data['event_list']:
#                 if event_mention['event_type'] not in events:
#                     events.append(event_mention['event_type'])
#                 for argument_mention in event_mention['arguments']:
#                     if argument_mention['role'] not in roles:
#                         roles.append(argument_mention['role'])
#                     if len(argument_mention['argument']) > L:
#                         L = len(argument_mention['argument'])
#                     if argument_mention['argument_start_index'] in Entity:
#                         if argument_mention['argument'] != Entity[argument_mention['argument_start_index']]:
#                             print(sentence)
#                     if argument_mention['argument_start_index'] not in Entity:
#                         Entity[argument_mention['argument_start_index']] =  argument_mention['argument']
#
#
# print(L)
# triggers = []
# for trigger in events:
#     triggers.append(trigger.split('-')[-1])
#
# if len(events) == len(set(triggers)):
#     print('OJBK')
# print(roles)
# print(len(roles))
# print(len(events))
# print(triggers)
# print(len(triggers))
#
#
# # with open('./data/num.json', 'w') as f:
# #     json.dump(b, f, indent=2, ensure_ascii=False)

