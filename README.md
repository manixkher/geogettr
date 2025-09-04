# geogettr

Geogettr is a deep learning project for geolocation. The model predicts a geocell, which is then converted to a lat/lon coordinate pair. 

We utilise a ResNet50 neural network, pretrained on ImageNet as our model architecture. We train on the OSV5M dataset, but filtered to only map Europe. Images are sorted into geographic cells (geocells) arranged in a quadtree structure. 

