from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_preprocessing import DataPreprocessing
from cnnClassifier import logger

STAGE_NAME = "Data Preprocessing"


class DataPreprocessingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_preprocess_data()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.process_and_save_images()
        
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e    