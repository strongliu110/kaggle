import numpy as np
import pandas as pd
import os

from scripts.data_load import get_data_df, get_test_df
from scripts.label_encode import Label
from scripts.dataset import DataSet
from scripts.models import get_compile_model
from scripts.model import Model

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

    # 加载模型权值
    # model.load_weights(output_path + "weights.best_01-0.30.hdf5")

    # 测试
    test_path = os.path.join('/'.join(path), 'seedling/input/test/')
    test_df = get_test_df(test_path)

    dataset.load_test(test_df, input_shape)

    # 预测
    pred_Y = model.predict(dataset.test_datas)
    pred_label = label.decode(pred_Y)

    # 保存预测值
    res = {'file': test_df['Name'], 'species': pred_label}
    res = pd.DataFrame(res)
    res.to_csv(output_path + "result.csv", index=False)
