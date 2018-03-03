from scripts.data_load import *
from scripts.models import *
from scripts.model import *

if __name__ == "main":
    data_path = ""
    data_df = get_data_df(data_path)

    label = Label()
    data_df['Label'] = label.encode(data_df['Label'])

    train_df, val_df = get_train_val_df(data_df)

    input_shape = (200, 200, 3)
    compile_model = get_compile_model('basic', input_shape, 10, )

    for path in train_df['Path']:
        train_img = cv2.imread(path)
        train_df['Input'] = train_img

    model = Model(compile_model)
    model.fit(train_df, val_df, epochs=30)

    test_path = ''
    test_df = get_test_df(test_path)
    for path in test_df['Path']:
        test_img = cv2.imread(path)
        test_img /= 255
        test_img['Input'] = test_img

    pred_code = model.predict(test_df['Input'])
    pred_label = label.decode(pred_code)
