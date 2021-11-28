# mail-spam-filter

#### Yêu cầu đã cài đặt và cấu hình Apache Spark, Python phiên bản 3.7

Thêm các thư viện Spark vào PyCharm IDE:

1.	Settings -> Project: mail-spam-filter -> Project structure -> Add Content Root
2.	Chọn tất cả các file .zip có từ $SPARK_HOME/python/lib
3.	Chọn Apply và OK

Thêm các biến môi trường trong Run Configurations:

-	PYSPARK_PYTHON=/usr/bin/python3

-	PYSPARK_DRIVER_PYTHON=/usr/bin/python3

-	PYTHONPATH=$SPARK_HOME/python

Cung cấp dữ liệu:

- Dữ liệu training nospam_training.txt, spam_training.txt, cũng như dữ liệu thử nghiệm nospam_testing.txt, spam_testing.txt cần được để vào theo đường dẫn ../spam-datasets/*.txt
