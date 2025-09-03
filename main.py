from pipelines.data_pipeline import data_pipeline

if __name__ == "__main__":
    pipe = data_pipeline()
    pipe.run()