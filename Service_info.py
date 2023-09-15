import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, lit
from synapse.ml.cognitive import TextToSpeech

# Initialize Key Vault client
key_vault_url = "https://audiobook-vault.vault.azure.net/"
credential = DefaultAzureCredential()
client = SecretClient(vault_url=key_vault_url, credential=credential)

# Get service key and storage details from Key Vault
service_key = client.get_secret("cognitive-api-key").value
service_loc = "eastus"
storage_key = os.environ['AZURE_STORAGE_ACCOUNT_KEY']
storage_account = "audiobook777"
storage_container = "audiobooks"

# Initialize Spark
spark = SparkSession.builder \
    .appName("Audiobook Generation") \
    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.11.2,org.apache.hadoop:hadoop-azure:3.2.0") \
    .getOrCreate()

# Configure Spark to use your storage account
spark_key_setting = f"fs.azure.account.key.{storage_account}.blob.core.windows.net"
spark.sparkContext._jsc.hadoopConfiguration().set(spark_key_setting, storage_key)

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient(account_url=f"https://{storage_account}.blob.core.windows.net", credential=storage_key)
container_client = blob_service_client.get_container_client(storage_container)

# User Defined Function to generate audio filenames
@udf
def make_audio_filename(part):
    return f"wasbs://{storage_container}@{storage_account}.blob.core.windows.net/audiobooks/audio_{part}.wav"

# Initialize TextToSpeech object
tts = TextToSpeech() \
    .setSubscriptionKey(service_key) \
    .setTextCol("text") \
    .setLocation(service_loc) \
    .setErrorCol("error") \
    .setVoiceName("en-US-SteffanNeural") \
    .setOutputFileCol("filename")

# Loop through each segment file
segment_directory = "/Volumes/Big Brain/Audiobooks-Python/WhatShallIDo - segments/audiobook_segments"
for i in range(1, 64):  # We have 63 segments
    segment_file = os.path.join(segment_directory, f"segment_{i}.txt")
    print(f"Processing segment file: {segment_file}")
    
    # Read the text data
    df = spark.read.text(segment_file).repartition(10).withColumn("filename", make_audio_filename(lit(f"{i}")))
    
    # Generate the audio
    audio = tts.transform(df).cache()
    print(f"Audio generation for segment {i} completed")
    
    # Collect the DataFrame to trigger the action
    audio.collect()

# Stop Spark
spark.stop() 

#Figure out how to upload the audio files to Azure Storage
