import re
import os

special_tokens = ['<H>', '</H>', '<R>', '</R>', '<T>', '</T>', '<S>']

def load_data(args):
    print('Load data')
    train_source = open(os.path.join(args.input_dir, 'train.source')).readlines()
    train_target = open(os.path.join(args.input_dir, 'train.target')).readlines()
    val_source = open(os.path.join(args.input_dir, 'val.source')).readlines()
    val_target = open(os.path.join(args.input_dir, 'val.target')).readlines()
    test_source = open(os.path.join(args.input_dir, 'test_both.source')).readlines()
    test_target = open(os.path.join(args.input_dir, 'test_both.target')).readlines()
    
    if args.reorder:
        print('Reorder triples')
        print(train_source[-1])
        train_source = reorder_triples(train_source)
        val_source = reorder_triples(val_source)
        test_source = reorder_triples(test_source)
        print(train_source[-1])
    
    train_set = [{'input': s, 'target': t} for s, t in zip(train_source, train_target)]
    val_set = [{'input': s, 'target': t} for s, t in zip(val_source, val_target)]
    test_set = [{'input': s, 'target': t} for s, t in zip(test_source, test_target)] 
    
    return train_set, val_set, test_set

def evaluate(args, outputs, *xargs):
    data_dir = './data/webnlg/data/'
    try:
        data_split = 'test_both' if args.inference else 'val'
    except:
        data_split = 'val'
    pred_file = os.path.join(args.output_dir, 'pred.txt')
    
    with open(pred_file, 'w') as f:
        for line in outputs:
            f.write('{}\n'.format(convert_text(line)))

    bleu_info = eval_bleu(data_dir, pred_file, data_split)
    print("BLEU: %.3f" % float(bleu_info.split(",")[0].split("BLEU = ")[1]))
    try:
        meteor_info = eval_meteor_test_webnlg(data_dir, pred_file, data_split)
        print("METEOR: %.3f" % (float(meteor_info.split("Final score:")[1])*100))
    except:
        print("METEOR: error")
    try:
        chrf_info = eval_chrf_test_webnlg(data_dir, pred_file, data_split)
        print("CHRF: %.3f" % float(chrf_info.split("c6+w2-avgF2")[1]))
    except:
        print("CHRF: error")    

    return float(bleu_info.split(",")[0].split("BLEU = ")[1])

def reorder_triples(sequences):
    reordered_sequences = []
    for sequence in sequences:
        triples = sequence.strip().split('<H>')
        triples = [triple for triple in triples if triple != '']
        triples = [(triple.split('<R>')[0].strip(), triple.split('<R>')[1].split('<T>')[0].strip(), triple.split('<T>')[-1].strip()) for triple in triples]
        reorder = {}
        for triple in triples:
            if triple[0] not in reorder:
                reorder[triple[0]] = {triple[1]: [triple[2]]}
            else:
                if triple[1] not in reorder[triple[0]]:
                    reorder[triple[0]][triple[1]] = [triple[2]]
                else:
                    reorder[triple[0]][triple[1]].append(triple[2])
        reordered_sequence = ''
        for key, value in reorder.items():
            reordered_sequence += " <S> <H> {} </H> ".format(key)
            # reordered_sequence += " <H> {} ".format(key)
            for k, v in value.items():
                reordered_sequence += " <R> {} </R> <T> {} </T> ".format(k, ' ; '.join(v))
            reordered_sequence += ' </S> '
        #     reordered_sequence += "<R> {} <T> {} </T> </R> ".format(k, ' <S> '.join(v))
        #     reordered_sequence += '</H>'
        # reordered_sequences.append(reordered_sequence.replace('  <', ' <').strip())
        reordered_sequences.append(reordered_sequence.replace('>  <', '> <').strip())
    return reordered_sequences

def convert_text(text):
    # text = text.lower()
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text

def eval_meteor_test_webnlg(folder_data, pred_file, dataset):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"
    cmd_string = "java -jar " + folder_data_before + "/meteor-1.5.jar " + pred_file + " " \
                  + folder_data + "/" + dataset + ".target_eval_meteor -l en -norm -r 3 > " + pred_file.replace("txt", "meteor")
    os.system(cmd_string)
    meteor_info = open(pred_file.replace("txt", "meteor"), 'r').readlines()[-1].strip()
    return meteor_info


def eval_chrf_test_webnlg(folder_data, pred_file, dataset):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"
    cmd_string = "python " + folder_data_before + "/chrf++.py -H " + pred_file + " -R " \
                  + folder_data + "/" + dataset + ".target_eval_crf > " + pred_file.replace("txt", "chrf")
    os.system(cmd_string)
    chrf_info_1 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[1].strip()
    chrf_info_2 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[2].strip()
    return chrf_info_1 + " " + chrf_info_2

def eval_bleu(folder_data, pred_file, dataset):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"
    cmd_string = "perl " + folder_data_before + "/multi-bleu.perl -lc " + folder_data + "/" + dataset + ".target_eval " \
                  + folder_data + "/" + dataset + ".target2_eval " + folder_data + "/" + dataset + ".target3_eval < " \
                  + pred_file + " > " + pred_file.replace("txt", "bleu")

    os.system(cmd_string)
    try:
        bleu_info = open(pred_file.replace("txt", "bleu"), 'r').readlines()[0].strip()
    except:
        bleu_info = -1
    return bleu_info


def eval_bleu_sents_tok(pred_file, folder_data, dataset):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"
    cmd_string = "perl " + folder_data_before + "/tokenizer.perl -threads 4 -no-escape < " + pred_file + " > " +\
                 pred_file + "_tok"
    os.system(cmd_string)
    cmd_string = "perl " + folder_data_before + "/multi-bleu.perl -lc " + folder_data + "/" + dataset + ".target.tok"\
                 + " < " + pred_file + "_tok" + " > " + pred_file.replace("txt", "bleu_data")
    os.system(cmd_string)
    try:
        bleu_info_data = open(pred_file.replace("txt", "bleu_data"), 'r').readlines()[0].strip()
    except:
        bleu_info_data = 'no data'
    return bleu_info_data


def eval_meteor(ref_file, pred_file):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"

    cmd_string = "java -jar " + folder_data_before + "/meteor-1.5.jar " + pred_file + " " \
                  + ref_file + " > " + pred_file.replace("txt", "meteor")
    os.system(cmd_string)
    meteor_info = open(pred_file.replace("txt", "meteor"), 'r').readlines()[-1].strip()
    return meteor_info


def eval_chrf(ref_file, pred_file):

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

