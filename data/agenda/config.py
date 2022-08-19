import os
import json
import re

special_tokens = ['-rrb-', '-lrb-', '--', '<title>', '</title>', '<relationship>', '</relationship>', '<entity>', '</entity>', 'USED-FOR', 'CONJUNCTION', 'FEATURE-OF', 'PART-OF', 'COMPARE', 'EVALUATE-FOR', 'HYPONYM-OF']

def load_data(args):
    print('Load data')
    train_data = json.load(open(os.path.join(args.input_dir, 'unprocessed.training.json')))
    val_data = json.load(open(os.path.join(args.input_dir, 'unprocessed.dev.json')))
    test_data = json.load(open(os.path.join(args.input_dir, 'unprocessed.test.json')))

    train_data, val_data, test_data = serializer(train_data), serializer(val_data), serializer(test_data)
    
    train_set = [{'input': s, 'target': t} for s, t in train_data]
    val_set = [{'input': s, 'target': t} for s, t in val_data]
    test_set = [{'input': s, 'target': t} for s, t in test_data] 
    
    return train_set, val_set, test_set

def evaluate(args, outputs, *xargs):
    data_dir = './data/agenda/data/'
    try:
        data_split = 'test' if args.inference else 'val'
    except:
        data_split = 'val'
    pred_file = os.path.join(args.output_dir, 'pred.txt')
    
    with open(pred_file, 'w') as f:
        for line in outputs:
            f.write('{}\n'.format(convert_text(line)))

    bleu_info = eval_bleu(data_dir, pred_file, data_split)
    print("BLEU: %.3f" % float(bleu_info.split(",")[0].split("BLEU = ")[1]))
    try:
        meteor_info = eval_meteor(data_dir, pred_file)
        print("METEOR: %.3f" % (float(meteor_info.split("Final score:")[1])*100))
    except:
        print("METEOR: error")
    try:
        chrf_info = eval_chrf(data_dir, pred_file)
        print("CHRF: %.3f" % float(chrf_info.split("c6+w2-avgF2")[1]))
    except:
        print("CHRF: error")    

    return float(bleu_info.split(",")[0].split("BLEU = ")[1])

def serializer(data):
    serialized_data = []
    for item in data:
        source = "<title> {} </title>".format(item['title'])
        source += " <relationship>"
        for relation in item['relations']:
            source += " {} ;".format(relation)
        source += " </relationship> <entity>"
        entity_types = item['types'].split()
        try:
            assert len(entity_types) == len(item['entities'])
        except:
            continue
        for entity, type in zip(item['entities'], entity_types):
            # source += ' <{}> {} </{}>'.format(type, entity, type)
            source += " {} ( {} ) ;".format(entity, type[1:-1])
        source += " </entity>"

        target = item['abstract_og']
        serialized_data.append((source, target))
    return serialized_data

def convert_text(text):
    #return text
    text = text.lower()
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text


def eval_bleu(folder_data, pred_file, dataset):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"

    if dataset == 'val':
        dataset = 'dev'
    ref_file = os.path.join(folder_data, '{}-tgt.txt'.format(dataset))

    cmd_string = "perl " + folder_data_before + "/multi-bleu.perl -lc " + ref_file \
                 + " < " + pred_file + " > " + pred_file.replace("txt", "bleu_data")
    os.system(cmd_string)

    try:
        bleu_info_data = open(pred_file.replace("txt", "bleu_data"), 'r').readlines()[0].strip()
    except:
        bleu_info_data = 'no data'

    return bleu_info_data


def eval_meteor(folder_data, pred_file, dataset):
    if dataset == 'val':
        dataset = 'dev'
    ref_file = os.path.join(folder_data, '{}-tgt.txt'.format(dataset))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"

    cmd_string = "java -jar " + folder_data_before + "/meteor-1.5.jar " + pred_file + " " \
                  + ref_file + " > " + pred_file.replace("txt", "meteor")

    os.system(cmd_string)

    meteor_info = open(pred_file.replace("txt", "meteor"), 'r').readlines()[-1].strip()

    return meteor_info


def eval_chrf(folder_data, pred_file, dataset):
    if dataset == 'val':
        dataset = 'dev'
    ref_file = os.path.join(folder_data, '{}-tgt.txt'.format(dataset))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"

    cmd_string = "python " + folder_data_before + "/chrf++.py -H " + pred_file + " -R " \
                  + ref_file + " > " + pred_file.replace("txt", "chrf")

    os.system(cmd_string)

    try:
        chrf_info_1 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[1].strip()
        chrf_info_2 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[2].strip()
        chrf_data = chrf_info_1 + " " + chrf_info_2
    except:
        chrf_data = "no data"

    return chrf_data