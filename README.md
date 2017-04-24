# GREP RESTful API

## Overview

GREP (GRoup Emotion Parser) aims at detecting human faces within the image and estimates both individual-level and group-level happiness intensity is set and configured on a virtual server. Currently, GREP only offers two APIs:

1. POST /v1.0.0/predict
2. POST /v1.0.0/feedback

You need to send a request with the url of the image or a base-64 format of the image. You have the option to specify whether to return annotated faces and cropped thumbnail of each face in base64. More detailed API documentation could be found [here](https://grep.net.cn/doc).

## Demo Website: grep.net.cn

For demonstration purpose, an interactive web application GREP is built and running at [here](https://grep.net.cn). It is an end-to-end system that detects faces in the group image and estimates happiness intensity on both individual level and group level. This app uses the RESTful API to perform the estimation task. It allows you to have multiple options to upload an image, including directly using url, selecting a local image, or using the webcam to take the picture. After uploading the image, each detected face together with individual happiness estimation and group happiness estimation will show on the webpage. The app also provides the user with a way to give feedback, which will serve as a training sample and help improve the system in the future.
		
The web application is the so called single-page application (SPA). It is such a web application that fits on a single web page with the goal of providing a more fluid user experience similar to a desktop application. 
	
The front-end of the web application is built upon Angular.js, which is an open-source web application framework, powerful for building SPA.

## Underlying Technique
The underlying technique and principle behind GREP is a Random Recurrent Deep Ensemble (RRDE) based framework proposed by Yichen in his undergraduate thesis.

More detailed description of the RRDE in introduced [here]().

## Technologies Used

- [Python](https://www.python.org/)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [Flask](http://flask.pocoo.org/)
- [Flask-swagger](https://github.com/gangverk/flask-swagger)
- [Swagger-UI](https://github.com/swagger-api/swagger-ui)
- [Angular.js](http://angularjs.org)
- [OpenAPI Specification](https://github.com/OAI/OpenAPI-Specification/)
- [OpenCV](http://opencv.org/)
- [MongoDB](https://docs.mongodb.com/manual/introduction/)

In addition and by necessity, the GREP app uses [HTML5](https://en.wikipedia.org/wiki/HTML5)/[CSS3](https://en.wikipedia.org/wiki/Cascading_Style_Sheets#CSS_3) + [Javascript](https://en.wikipedia.org/wiki/JavaScript)/[jQuery](https://jquery.com/).
