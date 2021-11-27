import pyspark
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.types import *
from pyspark.sql import functions as function
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt


class EmailSpamFilter:

    # constructor
    def __init__(self, spark_conf, spark_session, spark_context):
        self.conf = spark_conf
        self.spark = spark_session
        self.sc = spark_context

    # Transformations return RDDs
    # Actions returns something to the driver node
    # Model training is an action

    def create_dataframe(self, path, type_df):
        """
        don't read into an RDD, read into a DataFrame
        each line of the data is an email --> 1 Feature
        """
        sql_context = SQLContext(self.sc)
        schema = StructType([
            StructField('message', StringType(), False)
        ])
        # sql_context so it knows how to do SQL on it
        df = sql_context.read \
            .format('csv') \
            .option('delimiter', '\n') \
            .load(path, schema=schema)

        # add column 'type' (either 'spam' or 'normal') for the label
        df = df.withColumn('type', function.lit(type_df))

        return df

    def extract_features(self, df):
        """
        turn the data you have into the data you want
        - specify what the feature vector looks like
        - specify what the labels look like
        - transform strings into numeric values (StringIndexer)
        same step for training and testing
        """
        label_indexer = StringIndexer(inputCol='type', outputCol='label').setHandleInvalid(
            'skip')  # necessary because of a bug
        message_indexer = StringIndexer(inputCol='message', outputCol='message_indexed').setHandleInvalid('skip')

        # take the column and create a feature vector out of it
        assembler = VectorAssembler(
            inputCols=['message_indexed'],
            outputCol='feature'
        )
        preprocessing_stages = [label_indexer, message_indexer, assembler]
        return preprocessing_stages

    def model_training(self, preprocessing_stages, train_df, classification):
        if classification.lower() == 'lr':
            classifier = LogisticRegression(labelCol='label', featuresCol='feature')
        elif classification.lower() == 'svm':
            classifier = LinearSVC(labelCol='label', featuresCol='feature')
        # svm is the default case
        else:
            classifier = LinearSVC(labelCol='label', featuresCol='feature')
        preprocessing_stages.append(classifier)
        stages = preprocessing_stages
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(train_df)
        return model

    def evaluate_model(self, model, test_df, classification):
        predictions = model.transform(test_df)
        evaluator = BinaryClassificationEvaluator()
        # sensitivity = true-positive rate (AUROC)
        # specificity = false-positive rate (AUPR)
        auroc = evaluator.evaluate(predictions, {evaluator.metricName: 'areaUnderROC'})  # like accuracy
        aupr = evaluator.evaluate(predictions, {evaluator.metricName: 'areaUnderPR'})

        print('For {classification} the AUROC is {auroc} and the AUPR is {aupr}'.format(auroc=auroc, aupr=aupr,
                                                                                        classification=classification))
        # SVM has no summary to plot
        if classification.lower() == 'lr':
            training_summary = model.stages[-1].summary  # get the model from stages
            roc = training_summary.roc.toPandas()
            plt.plot(roc['FPR'], roc['TPR'])
            plt.ylabel('False Positive Rate (specificity)')
            plt.xlabel('True Positive Rate (sensitivity)')
            plt.title('ROC Curve of the Logistic Regression')
            plt.show()
        pass

    def run_svm(self):
        # get training for svm data
        no_spam_df = mail_filter.create_dataframe(path='../spam-datasets/nospam_training.txt', type_df='normal')
        spam_df = mail_filter.create_dataframe(path='../spam-datasets/spam_training.txt', type_df='spam')
        # combine both as training data
        train_df = no_spam_df.union(spam_df)
        ps = mail_filter.extract_features(train_df)

        # train svm
        svm_model = mail_filter.model_training(preprocessing_stages=ps, train_df=train_df, classification='svm')

        # get test data sets for svm
        no_spam_test_df = mail_filter.create_dataframe(path='../spam-datasets/nospam_testing.txt',
                                                       type_df='normal')
        spam_test_df = mail_filter.create_dataframe(path='../spam-datasets/spam_testing.txt', type_df='spam')
        # combine both as test data
        test_df = no_spam_test_df.union(spam_test_df)
        # evaluate svm
        mail_filter.evaluate_model(model=svm_model, test_df=test_df, classification='svm')
        pass

    def run_lr(self):
        # get training data for logistic regression
        no_spam_lr_df = mail_filter.create_dataframe(path='../spam-datasets/nospam_training.txt',
                                                     type_df='normal')
        spam_lr_df = mail_filter.create_dataframe(path='../spam-datasets/spam_training.txt', type_df='spam')
        # combine both as training data
        train_lr_df = no_spam_lr_df.union(spam_lr_df)
        ps_lr = mail_filter.extract_features(train_lr_df)

        # train logistic regression
        lr_model = mail_filter.model_training(preprocessing_stages=ps_lr, train_df=train_lr_df, classification='lr')

        # get test data sets for logistic regression
        no_spam_test_lr_df = mail_filter.create_dataframe(path='../spam-datasets/nospam_testing.txt',
                                                          type_df='normal')
        spam_test_lr_df = mail_filter.create_dataframe(path='../spam-datasets/spam_testing.txt', type_df='spam')
        # combine both as test data
        test_lr_df = no_spam_test_lr_df.union(spam_test_lr_df)

        # evaluate logistic regression
        mail_filter.evaluate_model(model=lr_model, test_df=test_lr_df, classification='lr')
        pass


if __name__ == '__main__':
    # Spark session & context
    conf = pyspark.SparkConf().set('spark.driver.host', '127.0.0.1')
    spark = SparkSession.builder.master('local').getOrCreate()
    sc = spark.sparkContext

    mail_filter = EmailSpamFilter(spark_conf=conf, spark_session=spark, spark_context=sc)

    mail_filter.run_lr()
    mail_filter.run_svm()
    sc.stop()
