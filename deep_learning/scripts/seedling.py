from scripts.data_load import *
from scripts.models import *
from scripts.model import *
from scripts.dataset import *

if __name__ == "__main__":
    # 数据加载
    module_path = os.path.dirname(__file__)
    path = module_path.split("/")[:-2]
    data_path = os.path.join('/'.join(path), 'seedling/input/train/')
    data_df = get_data_df(data_path)

    # 标签编码
    label = Label()
    label_list = label.encode(data_df['Label'])
    data_df['EncodeLabel'] = label_list

    input_shape = (200, 200, 3)

    # def read_image(path):
    #     train_img = cv2.resize(cv2.imread(path), (10, 10))
    #     return train_img
    # train_df['Input'] = train_df.apply(lambda row: read_image(row.Path), axis=1)

    # 分割数据集
    dataset = DataSet()
    dataset.load_kfold_train_val(data_df, input_shape)

    # 创建模型
    compile_model = get_compile_model('basic', input_shape, len(label_list[0]))
    output_path = os.path.join('/'.join(path), 'seedling/output/')
    model = Model(compile_model, output_path)

    # 训练模型
    model.fit(dataset, epochs=1)

    test_path = os.path.join('/'.join(path), 'seedling/input/test/')
    test_df = get_test_df(test_path)

    test_raw_lst = []
    test_id_lst = []
    for path in test_df['Path']:
        test_img = cv2.resize(cv2.imread(path), (input_shape[0], input_shape[1]))
        test_img /= 255
        test_raw_lst.append(test_img.ravel())
        test_id_lst.append(path.split('/')[-1])

    # 预测
    pred_code = model.predict(test_raw_lst)
    pred_label = label.decode(pred_code)

    res = {'file': test_id_lst, 'species': pred_label}
    res = pd.DataFrame(res)
    res.to_csv(output_path + "result.csv", index=False)
