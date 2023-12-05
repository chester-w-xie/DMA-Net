Title:  TUT Urban Acoustic Scenes 2018, Development dataset

# TUT Urban Acoustic Scenes 2018, Development dataset

[Audio Research Group / Tampere University of Technology](http://arg.cs.tut.fi/)

Authors

- Toni Heittola (<toni.heittola@tut.fi>, <http://www.cs.tut.fi/~heittolt/>)
- Annamaria Mesaros (<annamaria.mesaros@tut.fi>, <http://www.cs.tut.fi/~mesaros/>)
- Tuomas Virtanen (<tuomas.virtanen@tut.fi>, <http://www.cs.tut.fi/~tuomasv/>)

Recording and annotation

- Ronal Bejarano Rodriguez
- Eemi Fagerlund
- Aino Koskimies
- Toni Heittola

## 1. Dataset

TUT Urban Acoustic Scenes 2018 development dataset consists of 10-seconds audio segments from 10 acoustic scenes: 

- Airport - `airport`
- Indoor shopping mall - `shopping_mall`
- Metro station - `metro_station`
- Pedestrian street - `street_pedestrian` 
- Public square - `public_square`
- Street with medium level of traffic - `street_traffic`
- Travelling by a tram - `tram`
- Travelling by a bus - `bus`
- Travelling by an underground metro - `metro`
- Urban park - `park`

Each acoustic scene has 864 segments (144 minutes of audio). The dataset contains in total 24 hours of audio. 

The dataset was collected in Finland by Tampere University of Technology between 02/2018 - 03/2018. The data collection has received funding from the European Research Council under the ERC Grant Agreement 637422 EVERYSOUND.

[![ERC](https://erc.europa.eu/sites/default/files/content/erc_banner-horizontal.jpg "ERC")](https://erc.europa.eu/)

### Preparation of the dataset

The dataset was recorded in six large European cities: Barcelona, Helsinki, London, Paris, Stockholm, and Vienna. For all acoustic scenes, audio was captured in multiple locations: different streets, different parks, different shopping malls. In each location, multiple 2-3 minute long audio recordings were captured in a few slightly different positions (2-4) within the selected location. Collected audio material was cut into segments of 10 seconds length. 

The equipment used for recording consists of a binaural [Soundman OKM II Klassik/studio A3](http://www.soundman.de/en/products/) electret in-ear microphone and a [Zoom F8](https://www.zoom.co.jp/products/handy-recorder/zoom-f8-multitrack-field-recorder) audio recorder using 48 kHz sampling rate and 24 bit resolution. During the recording, the microphones were worn by the recording person in the ears, and head movement was kept to minimum.

Post-processing of the recorded audio involves aspects related to privacy of recorded individuals, and possible errors in the recording process. The material was screened for content, and segments containing close microphone conversation were eliminated. Some interferences from mobile phones are audible, but are considered part of real-world recording process.

### Dataset statistics

The dataset is perfectly balanced at acoustic scene level, with very slight differences in the number of segments from each city.

#### Audio segments

| Scene class        | Segments | Barcelona | Helsinki | London   | Paris    | Stockholm | Vienna   |
| ------------------ | -------- | --------- | -------- | -------- | -------- | --------- | -------- |
| Airport            | 864      | 128       | 149      | 145      | 156      | 158       | 128      |
| Bus                | 864      | 144       | 144      | 144      | 144      | 144       | 144      |
| Metro              | 864      | 141       | 144      | 146      | 144      | 145       | 144      |
| Metro station      | 864      | 144       | 144      | 144      | 144      | 144       | 144      |
| Park               | 864      | 144       | 144      | 144      | 144      | 144       | 144      |
| Public square      | 864      | 144       | 144      | 144      | 144      | 144       | 144      |
| Shopping mall      | 864      | 144       | 144      | 144      | 144      | 144       | 144      |
| Street, pedestrian | 864      | 145       | 145      | 145      | 144      | 145       | 140      |
| Street, traffic    | 864      | 144       | 144      | 144      | 144      | 144       | 144      |
| Tram               | 864      | 143       | 145      | 144      | 144      | 144       | 144      |
| **Total**          | **8640** | **1421**  | **1447** | **1444** | **1452** | **1456**  | **1420** |

#### Recording locations

| Scene class        | Location | Barcelona | Helsinki | London | Paris  | Stockholm | Vienna |
| ------------------ | -------- | --------- | -------- | ------ | ------ | --------- | ------ |
| Airport            | 22       | 4         | 3        | 3      | 4      | 5         | 3      |
| Bus                | 36       | 4         | 4        | 7      | 11     | 6         | 4      |
| Metro              | 29       | 3         | 5        | 4      | 9      | 4         | 4      |
| Metro station      | 40       | 5         | 6        | 12     | 9      | 4         | 4      |
| Park               | 25       | 4         | 4        | 4      | 4      | 5         | 4      |
| Public_square      | 24       | 4         | 4        | 4      | 4      | 4         | 4      |
| Shopping mall      | 22       | 4         | 4        | 2      | 4      | 4         | 4      |
| Street, pedestrian | 28       | 7         | 4        | 4      | 5      | 4         | 4      |
| Street, traffic    | 25       | 4         | 4        | 5      | 4      | 4         | 4      |
| Tram               | 35       | 4         | 4        | 9      | 9      | 5         | 4      |
| **Total**          | **286**  | **43**    | **42**   | **54** | **63** | **45**    | **39** |

### File structure

```
dataset root
│   README.md				this file, markdown-format
│   README.html				this file, html-format
│   meta.csv				meta data, csv-format with a header row, [audio file (string)][tab][scene label (string)][tab][identifier (string)][tab][source_label (string)]
│
└───audio					8640 audio segments, 24-bit 48kHz stereo
│   │   airport-barcelona-0-0-a.wav		file naming convention: [scene label]-[city]-[location id]-[segment id]-[device id].wav
│   │   airport-barcelona-0-1-a.wav
│   │   airport-barcelona-0-3-a.wav
│   │   ...
│   │   airport-barcelona-1-17-a.wav
│   │   airport-barcelona-1-18-a.wav
│   │   ...
│
└───evaluation_setup		cross-validation setup, 1 fold
    │   fold1_train.txt		training file list, csv-format, [audio file (string)][tab][scene label (string)]
    │   fold1_test.txt 		testing file list, csv-format, [audio file (string)]
    │   fold1_evaluate.txt 	evaluation file list, fold1_test.txt with added ground truth, csv-format, [audio file (string)][tab][scene label (string)]  

```

## 2. Usage

The partitioning of the data was done based on the location of the original recordings. All segments recorded at the same location were included into a single subset - either **development dataset** or **evaluation dataset**. For each acoustic scene, 864 segments were included in the development dataset provided here. Evaluation dataset is provided separately.

### Training / test setup

A suggested training/test partitioning of the development set is provided in order to make results reported with this dataset uniform. The partitioning is done such that the segments recorded at the same location are included into the same subset - either training or testing. The partitioning is done aiming for a 70/30 ratio between the number of segments in training and test subsets while taking into account recording locations, and selecting the closest available option. 

The setup is provided with the dataset in the directory `evaluation_setup`. 

#### Statistics

| Scene class        | Train / Segments | Train / Locations | Test / Segments | Test / Locations |
| ------------------ | ---------------- | ----------------- | --------------- | ---------------- |
| Airport            | 599              | 15                | 265             | 7                |
| Bus                | 622              | 26                | 242             | 10               |
| Metro              | 603              | 20                | 261             | 9                |
| Metro station      | 605              | 28                | 259             | 12               |
| Park               | 622              | 18                | 242             | 7                |
| Public square      | 648              | 18                | 216             | 6                |
| Shopping mall      | 585              | 16                | 279             | 6                |
| Street, pedestrian | 617              | 20                | 247             | 8                |
| Street, traffic    | 618              | 18                | 246             | 7                |
| Tram               | 603              | 24                | 261             | 11               |
| **Total**          | **6122**         | **203**           | **2518**        | **83**           |

#### Training

`evaluation setup\fold1_train.txt`
: training file list (in csv-format)

Format:
    
    [audio file (string)][tab][scene label (string)]

#### Testing

`evaluation setup\fold1_test.txt`
: testing file list (in csv-format)

Format:
    [audio file (string)]

#### Evaluating

`evaluation setup\fold1_evaluate.txt`
: evaluation file list (in csv-format), same as `fold1_test.txt` but with additional reference information. These two files are provided separately to prevent contamination with ground truth when testing the system

Format: 

    [audio file (string)][tab][scene label (string)] 

### Custom setups

If not using the provided training/test setup, pay attention to the segments recorded at the same location. Location identifier can be found from `meta.csv` or from audio file names:

    [scene label]-[city]-[location id]-[segment id]-[device id].wav

Make sure that all files having **same location id** are placed on the same side of the evaluation. In this dataset, device id is always the same (`a`).

## 3. Changelog

**v1.0 / 2018-04-24**

* Initial commit

## 4. License

    Copyright (c) 2018 Tampere University of Technology and its licensors 
    All rights reserved.
    Permission is hereby granted, without written agreement and without license or royalty 
    fees, to use and copy the TUT Urban Acoustic Scenes 2018 (“Work”) described in this document 
    and composed of audio and metadata. This grant is only for experimental and non-commercial 
    purposes, provided that the copyright notice in its entirety appear in all copies of this Work, 
    and the original source of this Work, (Audio Research Group from Laboratory of Signal
    Processing at Tampere University of Technology), 
    is acknowledged in any publication that reports research using this Work.
    Any commercial use of the Work or any part thereof is strictly prohibited. 
    Commercial use include, but is not limited to:
    - selling or reproducing the Work
    - selling or distributing the results or content achieved by use of the Work
    - providing services by using the Work. 
    
    IN NO EVENT SHALL TAMPERE UNIVERSITY OF TECHNOLOGY OR ITS LICENSORS BE LIABLE TO ANY PARTY
    FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE
    OF THIS WORK AND ITS DOCUMENTATION, EVEN IF TAMPERE UNIVERSITY OF TECHNOLOGY OR ITS
    LICENSORS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
    TAMPERE UNIVERSITY OF TECHNOLOGY AND ALL ITS LICENSORS SPECIFICALLY DISCLAIMS ANY
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
    FITNESS FOR A PARTICULAR PURPOSE. THE WORK PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND 
    THE TAMPERE UNIVERSITY OF TECHNOLOGY HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, 
    UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
