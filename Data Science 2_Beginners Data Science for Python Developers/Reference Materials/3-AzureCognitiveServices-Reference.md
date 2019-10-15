
# Section 3: Azure Cognitive Services

Just as you created a web service that could consume data and return predictions, so there are many AI software-as-a-service (SaaS) offerings on the web that will return predictions or classifications based on data you supply to them. One family of these is Microsoft Azure Cognitive Services.

The advantage of using cloud-based services is that they provide cutting-edge models that you can access without having to train them. This can help accelerate both your exploration and use of ML.

Azure provides Cognitive Services APIs that can be consumed using Python to conduct image recognition, speech recognition, and text recognition, just to name a few. For the purposes of this notebook, we're going to look at using the Computer Vision API and the Text Analytics API.

First, we’ll start by obtaining a Cognitive Services API key. Note that you can get a free key for seven days, and then you'll be required to pay.

To learn more about pricing for Cognitive Services, see https://azure.microsoft.com/en-us/pricing/details/cognitive-services/

Browse to **Try Azure Cognitive Services** at https://azure.microsoft.com/en-us/try/cognitive-services/

1. Select **Vision API**.
2. Select **Computer Vision**.
3. Click **Get API key**.
4. If prompted for credentials, select **Free 7-day trial**.

Complete the above steps to also retrieve a Text Analytics API key from the Language APIs category. (You can also do this by scrolling down on the page with your API keys and clicking **Add** under the appropriate service.)

Once you have your API keys in hand, you're ready to start.

&gt; **Learning goal:** By the end of this part, you should have a basic comfort with accessing cloud-based cognitive services by API from a Python environment.

## Azure Cognitive Services Computer Vision

Computer vision is a hot topic in academic AI research and in business, medical, government, and environmental applications. We will explore it here by seeing firsthand how computers can tag and identify images.

The first step in using the Cognitive Services Computer Vision API is to create a client object using the ComputerVisionClient class.

Replace **ACCOUNT_ENDPOINT** with the account endpoint provided from the free trial. Replace **ACCOUNT_KEY** with the account key provided from the free trial.


```python
!pip install azure-cognitiveservices-vision-computervision
```


```python
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

# Get endpoint and key from environment variables
endpoint = 'ACCOUNT_ENDPOINT'
# Example: endpoint = 'https://westcentralus.api.cognitive.microsoft.com'
key = 'ACCOUNT_KEY'
# Example key = '1234567890abcdefghijklmnopqrstuv

# Set credentials
credentials = CognitiveServicesCredentials(key)

# Create client
client = ComputerVisionClient(endpoint, credentials)
```

Now that we have a client object to work with, let's see what we can do.

Using analyze_image, we can see the properties of the image with VisualFeatureTypes.tags.


```python
url = 'https://cdn.pixabay.com/photo/2014/05/02/23/54/times-square-336508_960_720.jpg'

image_analysis = client.analyze_image(url,visual_features=[VisualFeatureTypes.tags])

for tag in image_analysis.tags:
    print(tag)
```

### Exercise:


```python
# How can you use the code above to also see the description using VisualFeatureTypes property?
```

Now let's look at the subject domain of the image. An example of a domain is celebrity.
As of now, the analyze_image_by_domain method only supports celebrities and landmarks domain-specific models.


```python
# This will list the available subject domains 
models = client.list_models()

for x in models.models_property:
    print(x)
```

Let's analyze an image by domain:


```python
# Type of prediction
domain = "landmarks"

# Public-domain image of Seattle
url = "https://images.pexels.com/photos/37350/space-needle-seattle-washington-cityscape.jpg"

# English-language response
language = "en"

analysis = client.analyze_image_by_domain(domain, url, language)

for landmark in analysis.result["landmarks"]:
    print(landmark["name"])
    print(landmark["confidence"])
```

### Exercise:


```python
# How can you use the code above to predict an image of a celebrity?
# Using this image, https://images.pexels.com/photos/270968/pexels-photo-270968.jpeg?
# Remember that the domains were printed out earlier.
```

Let's see how we can get a text description of an image using the describe_image method. Use max_descriptions to retrieve how many descriptions of the image the API service can find. 


```python
domain = "landmarks"
url = "https://images.pexels.com/photos/726484/pexels-photo-726484.jpeg"
language = "en"
max_descriptions = 3

analysis = client.describe_image(url, max_descriptions, language)

for caption in analysis.captions:
    print(caption.text)
    print(caption.confidence)
```

### Exercise:


```python
# What other descriptions can be found with other images?
# What happens if you change the count of descriptions to output?

```

Let's say that the images contain text. How do we retrieve that information? There are two methods that need to be used for this type of call. Batch_read_file and get_read_operation_result. TextOperationStatusCodes is used to ensure that the batch_read_file call is completed before the text is read from the image. 


```python
# import models
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
import time

url = "https://images.pexels.com/photos/6375/quote-chalk-think-words.jpg"
mode = TextRecognitionMode.handwritten
raw = True
custom_headers = None
numberOfCharsInOperationId = 36

# Async SDK call
rawHttpResponse = client.batch_read_file(url, mode, custom_headers,  raw)

# Get ID from returned headers
operationLocation = rawHttpResponse.headers["Operation-Location"]
idLocation = len(operationLocation) - numberOfCharsInOperationId
operationId = operationLocation[idLocation:]

# SDK call
while True:
    result = client.get_read_operation_result(operationId)
    if result.status not in ['NotStarted', 'Running']:
        break
    time.sleep(1)

# Get data
if result.status == TextOperationStatusCodes.succeeded:
    for textResult in result.recognition_results:
        for line in textResult.lines:
            print(line.text)
            print(line.bounding_box)
```

### Exercise:


```python
# What other images with words can be analyzed?
```

You can find addition Cognitive Services demonstrations at the following URLs:
 - https://aidemos.microsoft.com/
 - https://github.com/microsoft/computerscience/blob/master/Events%20and%20Hacks/Student%20Hacks/hackmit/cogservices_demos/
 - https://azure.microsoft.com/en-us/services/cognitive-services/directory/

Images come in varying sizes, and there might be cases where you want to create a thumbnail of the image. For this, we need to install the Pillow library, which you can learn about at https://python-pillow.org/. Pillow is the PIL fork, or Python Imaging Library, which allows for image processing. 


```python
# Install Pillow
!pip install Pillow
```

Now that the Pillow library is installed, we will import the Image module and create a thumbnail from a provided image. (Once generated, you can find the thumbnail image in your project folder on Azure Notebooks.)


```python
# Pillow package
from PIL import Image

# IO package to create local image
import io

width = 50
height = 50
url = "https://images.pexels.com/photos/37350/space-needle-seattle-washington-cityscape.jpg"

thumbnail = client.generate_thumbnail(width, height, url)

for x in thumbnail:
    image = Image.open(io.BytesIO(x))

image.save('thumbnail.jpg')
```

&gt; **Takeaway:** In this subsection, you explored how to access computer-vision cognitive services by API. Specifically, you used tools to analyze and describe images that you submitted to these services.

## Azure Cognitive Services Text Analytics

Another area where cloud-based AI shines is text analytics. Like computer vision, identifying and pulling meaning from natural human languages is really the intersection of a lot of specialized disciplines, so using cloud services for it provides an economical means of tapping a lot of cognitive horsepower.

To prepare to use the Cognitive Services Text Analytics API, the requests library must be imported, along with the ability to print out JSON formats.


```python
import requests
# pprint is pretty print (formats the JSON)
from pprint import pprint
from IPython.display import HTML
```

Replace 'ACCOUNT_KEY' with the API key that was created during the creation of the seven-day free trial account.


```python
subscription_key = 'ACCOUNT_KEY'
assert subscription_key

# If using a Free Trial account, this URL does not need to be udpated.
# If using a paid account, verify that it matches the region where the 
# Text Analytics Service was setup.
text_analytics_base_url = "https://westcentralus.api.cognitive.microsoft.com/text/analytics/v2.1/"
```

### Text Analytics API

Now it's time to start processing some text languages.

To verify the URL endpoint for text_analytics_base_url, run the following:


```python
language_api_url = text_analytics_base_url + "languages"
print(language_api_url)
```

The API requires that the payload be formatted in the form of documents containing `id` and `text` attributes:


```python
documents = { 'documents': [
    { 'id': '1', 'text': 'This is a document written in English.' },
    { 'id': '2', 'text': 'Este es un documento escrito en Español.' },
    { 'id': '3', 'text': '这是一个用中文写的文件' },
    { 'id': '4', 'text': 'Ez egy magyar nyelvű dokumentum.' },
    { 'id': '5', 'text': 'Dette er et dokument skrevet på dansk.' },
    { 'id': '6', 'text': 'これは日本語で書かれた文書です。' }
]}
```

The next lines of code call the API service using the requests library to determine the languages that were passed in from the documents:


```python
headers   = {"Ocp-Apim-Subscription-Key": subscription_key}
response  = requests.post(language_api_url, headers=headers, json=documents)
languages = response.json()
pprint(languages)
```

The next line of code outputs the documents in a table format with the language information for each document:


```python
table = []
for document in languages["documents"]:
    text  = next(filter(lambda d: d["id"] == document["id"], documents["documents"]))["text"]
    langs = ", ".join(["{0}({1})".format(lang["name"], lang["score"]) for lang in document["detectedLanguages"]])
    table.append("<tr><td>{0}</td><td>{1}</td>".format(text, langs))
HTML("<table><tr><th>Text</th><th>Detected languages(scores)</th></tr>{0}</table>".format("\n".join(table)))
```

The service did a pretty good job of identifying the languages. It did confidently identify the Danish phrase as being Norwegian, but in fairness, even linguists argue as to whether Danish and Norwegian constitute distinct languages or are dialects of the same language. (**Note:** Danes and Norwegians have no doubts on the subject.)

### Exercise:


```python
# Create another document set of text and use the text analytics API to detect the language for the text. 
```

### Sentiment Analysis API

Now that we know how to use the Text Analytics API to detect the language, let's use it for sentiment analysis. Basically, the computers at the other end of the API connection will judge the sentiments of written phrases (anywhere on the spectrum of positive to negative) based solely on the context clues provided by the text.


```python
# Verify the API URl source for the Sentiment Analysis API
sentiment_api_url = text_analytics_base_url + "sentiment"
print(sentiment_api_url)
```

As above, the Sentiment Analysis API requires the language to be passed in as documents with `id` and `text` attributes.


```python
documents = {'documents' : [
  {'id': '1', 'language': 'en', 'text': 'I had a wonderful experience! The rooms were wonderful and the staff was helpful.'},
  {'id': '2', 'language': 'en', 'text': 'I had a terrible time at the hotel. The staff was rude and the food was awful.'},  
  {'id': '3', 'language': 'es', 'text': 'Los caminos que llevan hasta Monte Rainier son espectaculares y hermosos.'},  
  {'id': '4', 'language': 'es', 'text': 'La carretera estaba atascada. Había mucho tráfico el día de ayer.'}
]}
```

Let's analyze the text using the Sentiment Analysis API to output a sentiment analysis score:


```python
headers   = {"Ocp-Apim-Subscription-Key": subscription_key}
response  = requests.post(sentiment_api_url, headers=headers, json=documents)
sentiments = response.json()
pprint(sentiments)
```

### Exercise:


```python
# Create another document set with varying degree of sentiment and use the Sentiment Analysis API to detect what
# the sentiment is
```

### Key Phrases API

We've detected the language type using the Text Analytics API and the sentiment using the Sentiment Analysis API. What if we want to detect key phrases in the text? We can use the Key Phrase API.


```python
# As with the other services, setup the Key Phrases API with the following parameters
key_phrase_api_url = text_analytics_base_url + "keyPhrases"
print(key_phrase_api_url)
```

Create the documents needed to pass to the Key Phrases API with the `id` and `text` attributes.


```python
documents = {'documents' : [
  {'id': '1', 'language': 'en', 'text': 'I had a wonderful experience! The rooms were wonderful and the staff was helpful.'},
  {'id': '2', 'language': 'en', 'text': 'I had a terrible time at the hotel. The staff was rude and the food was awful.'},  
  {'id': '3', 'language': 'es', 'text': 'Los caminos que llevan hasta Monte Rainier son espectaculares y hermosos.'},  
  {'id': '4', 'language': 'es', 'text': 'La carretera estaba atascada. Había mucho tráfico el día de ayer.'}
]}
```

Now, call the Key Phrases API with the formatted documents to retrieve the key phrases.


```python
headers   = {'Ocp-Apim-Subscription-Key': subscription_key}
response  = requests.post(key_phrase_api_url, headers=headers, json=documents)
key_phrases = response.json()
pprint(key_phrases)
```

We can make this easier to read by outputing the documents in an HTML table format.


```python
table = []
for document in key_phrases["documents"]:
    text    = next(filter(lambda d: d["id"] == document["id"], documents["documents"]))["text"]    
    phrases = ",".join(document["keyPhrases"])
    table.append("<tr><td>{0}</td><td>{1}</td>".format(text, phrases))
HTML("<table><tr><th>Text</th><th>Key phrases</th></tr>{0}</table>".format("\n".join(table)))
```

Now call the Key Phrases API with the formatted documents to retrive the key phrases. 

### Exercise:


```python
# What other key phrases can you come up with for analysis?
```

### Entities API

The final API we will use in the Text Analytics API service is the Entities API. This will retrieve attributes for documents provided to the API service.


```python
# Configure the Entities URI
entity_linking_api_url = text_analytics_base_url + "entities"
print(entity_linking_api_url)
```

The next step is creating a document with id and text attributes to pass on to the Entities API. 


```python
documents = {'documents' : [
  {'id': '1', 'text': 'Microsoft is an It company.'}
]}
```

Finally, call the service using the rest call below to retrieve the data listed in the text attribute.


```python
headers = {"Ocp-Apim-Subscription-Key": subscription_key}
response = requests.post(entity_linking_api_url, headers=headers, json=documents)
entities = response.json()
entities
```

### Exercise:


```python
# What other entities can be retrieved with the API?
# Create a document setup and use the Text Analytics, Sentiment Analysis, 
# Key Phrase, and Entities API services to retrieve the data.

```

&gt; **Takeaway:** In this subsection, you explored text analytics in the cloud. Specifically, you used a variety of different APIs to extract different information from text: language, sentiment, key phrases, and entities.

That's it the instructional portion of this course. In these eight sections, you've now seen the range of tools that go into preparing data for analysis and performing ML and AI analysis on data. In the next, concluding section, you will bring these skills together in a final project.
