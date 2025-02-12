# ai-fire-detection

## Check-In Times
1. Jan 31 Slot 2
2. Feb 12 Slot 1
3. Feb 26 Slot 2
4. March 12 Slot 1
5. March 24 Slot 2
6. April 7 Slot 1

## Useful Links
- [Shared Google Drive](https://drive.google.com/drive/folders/1WcSBmiSUflXx6g88-UZEjkvhA1224XAy?usp=drive_link)
- [Kanban Board](https://vanderbilt365-my.sharepoint.com/personal/daniel_moyer_vanderbilt_edu/Lists/Fire%20Detection/AllItems.aspx?viewid=abb20dc8-8d62-4e84-b0cc-a5e494e4a15d&sw=bypass&bypassReason=abandoned&e=3:0c4369ef1e8248118e15dbaeb750a469&sharingv2=true&fromShare=true&at=9&CID=280d7da1-90d2-7000-b8e9-d5036e1da760&cidOR=SPO)

## Efficient and Compact Convolutional Neural Network Architectures for Non-temporal Real-time Fire Detection
Source: https://github.com/duncangrimes/ai-fire-detection (available under MIT license)

### Instructions to run inference using pre-trained models:

1. Install **pytorch >= 1.5.0** with torchvision

2. Install the requirements

    ~~~
    pip3 install -r requirements.txt
    ~~~
  
3. Download pre-trained models ([**nasnetonfire/shufflenetonfire**](https://collections.durham.ac.uk/downloads/r1tb09j570z)) in ```./weights``` directory and test video in ```./demo``` directory as follows:

    ~~~
    sh ./download-models.sh
    ~~~

  This download script (```download-models.sh```) will create an additional ```weights``` directory containing the pre-trained models and ```demo``` directory containing a test video file.

4. To run {fire, no-fire} classification on **full-frame**:

    ~~~
    python3 simple_inference.py
    --image test_images/paper.jpg
    --model shufflenetonfire
    --weight weights/shufflenet_ff.pt
    ~~~

## References:

[Efficient and Compact Convolutional Neural Network Architectures for Non-temporal Real-time Fire Detection](https://breckon.org/toby/publications/papers/thompson20fire.pdf)
(Thomson, Bhowmik, Breckon), In Proc. International Conference on Machine Learning Applications, IEEE, 2020.
```
@InProceedings{thompson20fire,
  author = {Thompson, W. and Bhowmik, N. and Breckon, T.P.},
  title = {Efficient and Compact Convolutional Neural Network Architectures for Non-temporal Real-time Fire Detection},
  booktitle = {Proc. Int. Conf. Machine Learning Applications},
  pages = {136-141},
  year = {2020},
  month = {December},
  publisher = {IEEE},
  keywords = {fire detection, CNN, deep-learning real-time, neural architecture search, nas, automl, non-temporal},
  url = {http://breckon.org/toby/publications/papers/thompson20fire.pdf},
  doi = {10.1109/ICMLA51294.2020.00030},
  arxiv = {http://arxiv.org/abs/2010.08833},
}