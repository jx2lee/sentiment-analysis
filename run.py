from utils import set_logger
from model_utils import *
from model import *
import fire


def main(token_type, model_type, num_layers, checkpoint_path):
    logger = set_logger('run')
    if token_type == "phoneme":
        x_train, x_test, y_train, y_test = load_data(np_path="output/npz", x_name="x_phoneme.npz",
                                                     y_name="y_phoneme.npz")
        df_train = array_to_df(x_train, y_train)
        df_test = array_to_df(x_test, y_test)
        train_set = Data(df_train)
        test_set = Data(df_test)

        model = build_phoneme_model(model_type=model_type, dict_size=53, state_size=128, batch_size=100)
        logger.info(f"Succeeded Creating Model:\t{model}")

        train_phoneme_model(graph=model, train_set=train_set, test_set=test_set, checkpoint_path=checkpoint_path,
                            save_df_name="result_phoneme.pkl", save_time_name="result_learning_time.pkl")
        logger.info(f"Run Train model..")

    logger.info(f'Finished training model\ttoken_type: {token_type}\tmodel_type: {model_type}\t'
                f'number_of_layers: {num_layers}\tcheckpoint_path: {checkpoint_path}')


if __name__ == '__main__':
    fire.Fire(main)
