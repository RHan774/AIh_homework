import my_utils


config = my_utils.read_config("DataProcess")

"""回归拟合中，可修改的数据集参数"""
config_regression = config["Regression"]
train_num = config_regression["train_num"]
validate_num = config_regression["validate_num"]
test_num = config_regression["test_num"]
noise_std = config_regression["noise_std"]
regression_datapath = config["Regression"]["data_path"]

"""分类问题中，可修改的数据集参数"""
config_classifier = config["Classifier"]
images_addr = config_classifier["images_addr"]
labels_addr = config_classifier["labels_addr"]
train_num = config_classifier["train_num"]
validate_num = config_classifier["validate_num"]
flatten = config_classifier["flatten"]
one_hot = config_classifier["one_hot"]
classifier_datapath = config_classifier["data_path"]

test_images = config_classifier["test_images"]
test_labels = config_classifier["test_labels"]
test_num = config_classifier["test_num"]

# my_utils.generate_sin_data(train_num = train_num, validate_num=validate_num, test_num=test_num,
#                            noise_std=noise_std, data_path=regression_datapath)

# my_utils.save_parse_data(images_addr=images_addr, labels_addr=labels_addr, train_num=train_num,
#                          validate_num=validate_num, flatten=flatten, one_hot=one_hot, data_path=classifier_datapath)

my_utils.save_test_data(images_addr=test_images, labels_addr=test_labels, test_num=test_num,
                        flatten=flatten, one_hot=one_hot, data_path=classifier_datapath)