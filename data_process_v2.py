import glob
import json
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer


# from zhconv import convert
def process_wiki_clean():
    with open('/mnt/pfs/data_team/maoxiangyi/wikipedia-cn-20230720-filtered/wikipedia-cn-20230720-filtered.json', 'r',
              encoding='utf-8') as f:
        data = json.load(f)
    doc_ids = []
    for line in tqdm(data):
        text = line['completion']
        text_id = tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id) > 5:
            doc_ids += text_id
    arr = np.array(doc_ids, dtype=np.uint16)
    with open('/mnt/pfs/data_team/maoxiangyi/data/wiki.bin', 'wb') as f:
        f.write(arr.tobytes())


def process_medical(data_path, name):
    f = open(data_path, 'r', encoding='utf-8')
    doc_ids = []
    while True:
        line = f.readline()
        if not line:
            break
        line = json.loads(line)
        text = line['text']
        text_id = tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id) > 5:
            doc_ids += text_id
    arr = np.array(doc_ids, dtype=np.uint16)
    with open('./data/medical_{}.bin'.format(name), 'wb') as f:
        f.write(arr.tobytes())


def sft_to_pretrain():
    doc_ids = []

    '''
    df=pd.read_csv('./data/medical_qa_144w.csv')
    for _,q,a in tqdm(df.itertuples()):
        q_id = tokenizer.encode(q,add_special_tokens=False)
        a_id = tokenizer.encode(a,add_special_tokens=False)
        #
        print(q)
        print(a)
        print('-----')
        text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
        if len(text_id)>5:
            doc_ids+=text_id
    '''

    with open('./data/shibing624_medical/finetune/train_en_1.json', 'r', encoding='utf-8') as f:
        for row in f:
            line = json.loads(row)
            q = line['input']
            a = line['output']
            q_id = tokenizer.encode(q, add_special_tokens=False)
            a_id = tokenizer.encode(a, add_special_tokens=False)
            text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
            if len(text_id) > 5:
                doc_ids += text_id
    with open('./data/shibing624_medical/finetune/test_en_1.json', 'r', encoding='utf-8') as f:
        for row in f:
            line = json.loads(row)
            q = line['input']
            a = line['output']
            q_id = tokenizer.encode(q, add_special_tokens=False)
            a_id = tokenizer.encode(a, add_special_tokens=False)
            text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
            if len(text_id) > 5:
                doc_ids += text_id
    with open('./data/shibing624_medical/finetune/valid_en_1.json', 'r', encoding='utf-8') as f:
        for row in f:
            line = json.loads(row)
            q = line['input']
            a = line['output']
            q_id = tokenizer.encode(q, add_special_tokens=False)
            a_id = tokenizer.encode(a, add_special_tokens=False)
            text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
            if len(text_id) > 5:
                doc_ids += text_id

    with open('./data/shibing624_medical/finetune/train_zh_0.json', 'r', encoding='utf-8') as f:
        for row in f:
            line = json.loads(row)
            q = line['instruction'] + line['input']
            a = line['output']
            q_id = tokenizer.encode(q, add_special_tokens=False)
            a_id = tokenizer.encode(a, add_special_tokens=False)
            text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
            if len(text_id) > 5:
                doc_ids += text_id
    with open('./data/shibing624_medical/finetune/test_zh_0.json', 'r', encoding='utf-8') as f:
        for row in f:
            line = json.loads(row)
            q = line['instruction'] + line['input']
            a = line['output']
            q_id = tokenizer.encode(q, add_special_tokens=False)
            a_id = tokenizer.encode(a, add_special_tokens=False)
            text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
            if len(text_id) > 5:
                doc_ids += text_id
    with open('./data/shibing624_medical/finetune/valid_zh_0.json', 'r', encoding='utf-8') as f:
        for row in f:
            line = json.loads(row)
            q = line['instruction'] + line['input']
            a = line['output']
            q_id = tokenizer.encode(q, add_special_tokens=False)
            a_id = tokenizer.encode(a, add_special_tokens=False)
            text_id = q_id + a_id + [tokenizer.special_tokens['<eos>']]
            if len(text_id) > 5:
                doc_ids += text_id

    arr = np.array(doc_ids, dtype=np.uint16)
    print(arr.shape)
    with open('./data/medical_qa.bin', 'wb') as f:
        f.write(arr.tobytes())


def process_baidu():
    BATCH_SIZE = 1000000

    cnt = 0
    batch_cnt = 0
    token = 0
    doc_ids = []

    f1 = open('/mnt/pfs/data_team/maoxiangyi/BaiduBaike-5.63M/563w_baidubaike.json', 'r', encoding='utf-8')

    while True:
        line = f1.readline()
        if not line:
            break
        line = json.loads(line)
        text = ''
        try:
            text += line['title'] + '：' + line['summary']
        except:
            pass
        for per in line['sections']:
            text += per['title'] + '：' + per['content'] + '。'
        text_id = tokenizer.encode(text, add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id) > 5:
            doc_ids += text_id
        cnt += 1
        if cnt % BATCH_SIZE == 0:
            batch_cnt += 1
            arr = np.array(doc_ids, dtype=np.uint16)
            doc_ids = []
            print('cnt:', cnt, 'arr_shape:', arr.shape)
            with open('/mnt/pfs/data_team/maoxiangyi/data/baidubaike_563w_{}.bin'.format(batch_cnt), 'wb') as f2:
                f2.write(arr.tobytes())
            del arr

    if not doc_ids:
        batch_cnt += 1
        arr = np.array(doc_ids, dtype=np.uint16)
        print('cnt:', cnt, 'arr_shape:', arr.shape)
        with open('/mnt/pfs/data_team/maoxiangyi/data/baidubaike_563w_{}.bin'.format(batch_cnt), 'wb') as f:
            f.write(arr.tobytes())


def process_c4():
    c4_zh_paths = glob.glob('/mnt/pfs/data_team/maoxiangyi/chinese-c4/data/*')
    c4_zh_paths = sorted(c4_zh_paths)
    print(len(c4_zh_paths))
    cnt = 0
    token = 0
    doc_ids = []
    for per in tqdm(c4_zh_paths[54:]):
        file_name = per.split('/')[-1]
        with open(per, 'r') as f:
            for line in f:
                text = json.loads(line)
                text = text['text']
                text_id = tokenizer.encode(text, add_special_tokens=False)
                text_id.append(tokenizer.special_tokens['<eos>'])
                if len(text_id) > 5:
                    doc_ids += text_id
                cnt += 1

        arr = np.array(doc_ids, dtype=np.uint16)
        with open('/mnt/pfs/data_team/maoxiangyi/data/c4_zh_{}.bin'.format(file_name), 'wb') as f:
            f.write(arr.tobytes())
            f.flush()
        doc_ids = []
        print(arr.shape)


def process_file(per):
    file_name = per.split('/')[-1]
    print(file_name)
    doc_ids = []
    with open(per, 'r') as f:
        data = json.load(f)
        for text in data:
            text = text['title'] + text['content']
            text_id = tokenizer.encode(text, add_special_tokens=False)
            text_id.append(tokenizer.special_tokens['<eos>'])
            if len(text_id) > 5:
                doc_ids += text_id

    arr = np.array(doc_ids, dtype=np.uint16)
    with open('/mnt/pfs/data_team/maoxiangyi/data/wudao200_{}.bin'.format(file_name), 'wb') as f:
        f.write(arr.tobytes())
        f.flush()
    doc_ids = None


def process_wudao():
    wudao_zh_paths = glob.glob('/mnt/pfs/data_team/maoxiangyi/WuDaoCorpus2.0_base_200G/*')
    wudao_zh_paths = sorted(wudao_zh_paths)
    print(len(wudao_zh_paths))

    filtered_paths = [path for path in wudao_zh_paths if not path.endswith((
        'part-2021012611.json', 'part-2021012612.json', 'part-2021012613.json',
        'part-2021012614.json', 'part-2021012615.json', 'part-2021012616.json',
        'part-2021012617.json', 'part-2021012618.json', 'part-2021012619.json',
        'part-2021012620.json', 'part-2021012621.json', 'part-2021012622.json',
        'part-2021012623.json', 'part-2021012700.json', 'part-2021012701.json',
        'part-2021012702.json', 'part-2021012703.json', 'part-2021012704.json',
        'part-2021012705.json', 'part-2021012706.json', 'part-2021012707.json',
        'part-2021012709.json', 'part-2021012710.json', 'part-2021012711.json',
        'part-2021012712.json', 'part-2021012713.json', 'part-2021012714.json',
        'part-2021012715.json', 'part-2021012716.json', 'part-2021012717.json',
        'part-2021012718.json', 'part-2021012719.json', 'part-2021012720.json',
        'part-2021012721.json', 'part-2021012722.json', 'part-2021012723.json',
        'part-2021012800.json', 'part-202101281a.json', 'part-202101281b.json',
        'part-202101281c.json'))]

    filtered_paths = [path for path in filtered_paths if not path.endswith((
        'part-2021009337.json', 'part-2021011897.json', 'part-2021012501.json',
        'part-2021012502.json', 'part-2021012503.json', 'part-2021012504.json',
        'part-2021012505.json', 'part-2021012506.json', 'part-2021012507.json',
        'part-2021012508.json', 'part-2021012509.json', 'part-2021012510.json',
        'part-2021012511.json', 'part-2021012512.json', 'part-2021012513.json',
        'part-2021012514.json', 'part-2021012515.json', 'part-2021012516.json',
        'part-2021012517.json', 'part-2021012518.json', 'part-2021012519.json',
        'part-2021012520.json', 'part-2021012521.json', 'part-2021012522.json',
        'part-2021012523.json', 'part-2021012524.json', 'part-2021012525.json',
        'part-2021012526.json', 'part-2021012527.json', 'part-2021012528.json',
        'part-2021012612.json', 'part-2021012613.json', 'part-2021012614.json',
        'part-2021012615.json', 'part-2021012616.json', 'part-2021012617.json',
        'part-2021012618.json', 'part-2021012619.json', 'part-2021012620.json',
        'part-2021012621.json', 'part-2021012622.json', 'part-2021012623.json',
        'part-2021012700.json', 'part-2021012701.json', 'part-2021012702.json',
        'part-2021012703.json', 'part-2021012704.json', 'part-2021012705.json',
        'part-2021012706.json', 'part-2021012707.json', 'part-2021012709.json',
        'part-2021012710.json', 'part-2021012711.json', 'part-2021012712.json'))]

    filtered_paths = [path for path in filtered_paths if not path.endswith(('part-2021009337.json','part-2021011897.json','part-2021012501.json','part-2021012502.json','part-2021012503.json','part-2021012504.json','part-2021012505.json','part-2021012506.json','part-2021012507.json','part-2021012508.json','part-2021012509.json','part-2021012510.json','part-2021012511.json','part-2021012512.json','part-2021012513.json','part-2021012514.json','part-2021012515.json','part-2021012516.json','part-2021012517.json','part-2021012518.json','part-2021012519.json','part-2021012520.json','part-2021012521.json','part-2021012522.json','part-2021012523.json','part-2021012524.json','part-2021012525.json','part-2021012526.json','part-2021012527.json','part-2021012528.json','part-2021012612.json','part-2021012613.json','part-2021012614.json','part-2021012615.json','part-2021012616.json','part-2021012617.json','part-2021012618.json','part-2021012619.json','part-2021012620.json','part-2021012621.json','part-2021012622.json','part-2021012623.json','part-2021012700.json','part-2021012701.json','part-2021012702.json','part-2021012703.json','part-2021012704.json','part-2021012705.json','part-2021012706.json','part-2021012707.json','part-2021012709.json','part-2021012710.json','part-2021012711.json','part-2021012712.json','part-2021013086.json','part-2021013704.json','part-2021013835.json','part-2021014043.json','part-2021014084.json','part-2021014209.json','part-2021014529.json','part-2021014840.json','part-2021015007.json','part-2021015155.json','part-2021015165.json','part-2021015264.json','part-2021015294.json','part-2021015368.json','part-2021015411.json','part-2021015496.json','part-2021015592.json','part-2021015607.json','part-2021015661.json','part-2021016046.json','part-2021016146.json','part-2021016853.json','part-2021016902.json','part-2021016914.json','part-2021016918.json','part-2021016947.json','part-2021017003.json','part-2021017056.json','part-2021017254.json','part-2021017289.json','part-2021017304.json','part-2021017985.json','part-2021018009.json','part-2021018037.json','part-2021018495.json','part-2021019264.json','part-2021019376.json','part-2021019380.json','part-2021019390.json','part-2021019549.json','part-2021019733.json','part-2021019930.json','part-2021019940.json','part-2021019969.json','part-2021020035.json','part-2021020076.json','part-2021020098.json','part-2021020110.json','part-2021020123.json','part-2021020124.json','part-2021020127.json','part-2021020181.json','part-2021020302.json','part-2021020338.json','part-2021020385.json','part-2021020401.json','part-2021020421.json','part-2021020424.json','part-2021020428.json','part-2021020429.json','part-2021020443.json','part-2021020446.json','part-2021020610.json','part-2021020726.json','part-2021020765.json','part-2021020789.json','part-2021020809.json','part-2021020845.json','part-2021020867.json','part-2021020870.json','part-2021020871.json','part-2021020878.json','part-2021020907.json','part-2021020967.json','part-2021020979.json','part-2021020994.json','part-2021021303.json','part-2021021308.json','part-2021021323.json','part-2021021333.json','part-2021021336.json','part-2021021345.json','part-2021021428.json','part-2021021437.json','part-2021021441.json','part-2021021442.json','part-2021021645.json','part-2021021707.json','part-2021021710.json','part-2021021736.json','part-2021021738.json','part-2021021741.json','part-2021021745.json','part-2021021764.json','part-2021021781.json','part-2021021789.json','part-2021021792.json','part-2021021799.json','part-2021021808.json','part-2021021841.json','part-2021021861.json','part-2021021896.json','part-2021021905.json','part-2021021914.json','part-2021021921.json','part-2021021923.json','part-2021021924.json','part-2021021957.json','part-2021022016.json','part-2021022027.json','part-2021022028.json','part-2021022029.json','part-2021022031.json','part-2021022032.json','part-2021022050.json','part-2021022053.json','part-2021022055.json','part-2021022066.json','part-2021022069.json','part-2021022093.json','part-2021022097.json','part-2021022106.json','part-2021022128.json','part-2021022131.json','part-2021022142.json','part-2021022150.json','part-2021022163.json','part-2021022169.json','part-2021022198.json','part-2021022247.json','part-2021022308.json','part-2021022310.json','part-2021022328.json','part-2021022340.json','part-2021022349.json','part-2021022352.json','part-2021022364.json','part-2021022386.json','part-2021022395.json','part-2021022426.json','part-2021022428.json','part-2021022433.json','part-2021022454.json','part-2021022455.json','part-2021022463.json','part-2021022474.json','part-2021022478.json','part-2021022489.json','part-2021022522.json','part-2021022536.json','part-2021022567.json','part-2021022586.json','part-2021022591.json','part-2021022605.json','part-2021022607.json','part-2021022612.json','part-2021022618.json','part-2021022631.json','part-2021022637.json','part-2021022642.json','part-2021022649.json','part-2021022651.json','part-2021022652.json','part-2021022671.json','part-2021022678.json','part-2021022683.json','part-2021022693.json','part-2021022694.json','part-2021022698.json','part-2021022713.json','part-2021022737.json','part-2021022805.json','part-2021022810.json','part-2021022835.json','part-2021022838.json','part-2021022851.json','part-2021022852.json','part-2021022859.json','part-2021022870.json','part-2021022891.json','part-2021022906.json','part-2021022921.json','part-2021022957.json','part-2021022959.json','part-2021022967.json','part-2021022988.json','part-2021022991.json','part-2021022993.json','part-2021022996.json','part-2021023008.json','part-2021023014.json','part-2021023020.json','part-2021023049.json','part-2021023078.json','part-2021023102.json','part-2021023104.json','part-2021023106.json','part-2021023111.json','part-2021023123.json','part-2021023129.json','part-2021023142.json','part-2021023151.json','part-2021023162.json','part-2021023179.json','part-2021023203.json','part-2021023207.json','part-2021023213.json','part-2021023244.json','part-2021023245.json','part-2021023247.json','part-2021023275.json','part-2021023282.json','part-2021023294.json','part-2021023302.json','part-2021023329.json','part-2021023335.json','part-2021023350.json','part-2021023358.json','part-2021023389.json','part-2021023395.json','part-2021023396.json','part-2021023403.json','part-2021023418.json','part-2021023427.json','part-2021023451.json','part-2021023461.json','part-2021023478.json','part-2021023483.json','part-2021023489.json','part-2021023503.json','part-2021023507.json','part-2021023523.json','part-2021023529.json','part-2021023549.json','part-2021023553.json','part-2021023574.json','part-2021023581.json','part-2021023623.json','part-2021023626.json','part-2021023637.json','part-2021023640.json','part-2021023643.json','part-2021023649.json','part-2021023658.json','part-2021023667.json','part-2021023688.json','part-2021023735.json','part-2021023736.json','part-2021023738.json','part-2021023741.json','part-2021023747.json','part-2021023761.json','part-2021023780.json','part-2021023783.json','part-2021023807.json','part-2021023818.json','part-2021023823.json','part-2021023842.json','part-2021023855.json','part-2021023873.json','part-2021023879.json','part-2021023885.json','part-2021023912.json','part-2021023922.json','part-2021023945.json','part-2021023970.json','part-2021023975.json','part-2021023985.json','part-2021023988.json','part-2021024098.json','part-2021024105.json','part-2021024145.json','part-2021024151.json','part-2021024163.json','part-2021024167.json','part-2021024186.json','part-2021024402.json','part-2021024454.json','part-2021024489.json','part-2021024532.json','part-2021024535.json','part-2021024569.json','part-2021024632.json','part-2021024638.json','part-2021024703.json','part-2021024803.json','part-2021024831.json','part-2021024834.json','part-2021024878.json','part-2021264921.json','part-2021270707.json','part-2021271152.json','part-2021274126.json','part-2021276655.json','part-2021277544.json','part-2021278643.json'))]

    for per in tqdm(filtered_paths):
        print(per)
        process_file(per)

if __name__=="__main__":
    tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
    # 数据预处理-如果下载分词处理后的数据，可以不用执行以下函数
    #process_wiki_clean()
    # process_medical('./data/shibing624_medical/pretrain/medical_book_zh.json','book')
    # process_medical('./data/shibing624_medical/pretrain/train_encyclopedia.json','encyclopedia')
    #process_baidu()
    #process_c4()
    # process_wudao()

    # print('data processing finished!')

    # # 分词处理后的文件列表
    # data_path_list=[
    #     './data/baidubaike_563w_1.bin',
    #     './data/baidubaike_563w_2.bin',
    #     './data/baidubaike_563w_3.bin',
    #     './data/baidubaike_563w_4.bin',
    #     './data/baidubaike_563w_5.bin',
    #     './data/medical_book.bin',
    #     './data/medical_encyclopedia.bin',
    #     './data/wiki.bin',
    #     './data/c4_zh_0.bin',
    #     './data/c4_zh_1.bin',
    #     './data/c4_zh_2.bin',
    #     './data/c4_zh_3.bin',
    #     './data/c4_zh_4.bin',
    #     './data/c4_zh_5.bin',
    #     './data/c4_zh_6.bin',
    #     './data/c4_zh_7.bin',
    #     './data/c4_zh_8.bin',
    #     './data/wudaocorpus_zh_0.bin',
    #     './data/wudaocorpus_zh_1.bin',
    #     './data/wudaocorpus_zh_2.bin',
    #     './data/wudaocorpus_zh_3.bin',
    #     './data/wudaocorpus_zh_4.bin',
    #     './data/wudaocorpus_zh_5.bin',
    #     './data/wudaocorpus_zh_6.bin',
    #     './data/wudaocorpus_zh_7.bin',
    #     './data/wudaocorpus_zh_8.bin',
    #     './data/wudaocorpus_zh_9.bin',
    #     './data/wudaocorpus_zh_10.bin',
    #     './data/wudaocorpus_zh_11.bin',
    #     './data/wudaocorpus_zh_12.bin',
    #     './data/wudaocorpus_zh_13.bin',
    #     './data/wudaocorpus_zh_14.bin',
    #     './data/wudaocorpus_zh_15.bin',
    #     './data/wudaocorpus_zh_16.bin',
    # ]
    data_lst=[]
    for data_path in tqdm(data_path_list):
        with open(data_path,'rb') as f:
            data=np.fromfile(f,dtype=np.uint16)
            data_lst.append(data)
    arr = np.concatenate(data_lst)
    print(arr.shape)
    with open('./data/pretrain_data.bin','wb') as f:
        f.write(arr.tobytes())
