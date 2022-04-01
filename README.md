# Correlational Analysis: GPUs and Video Games

# Overview

GPUs are best known for its use in modern videos and games and gained popularity through video games.[3][4] "GPUs were originally designed to accelerate the rendering of 3D graphics. Over time, they became more flexible ... This allowed graphics programmers to create more interesting visual effects and realistic scenes with advanced lighting and shadowing techniques."[5] 
A market for GPUs was needed in the form of video games since GPUs provided acceleration of rendering 3D graphics which boosted the popularity of GPUs in the earlier years. Video games still have a major impact on the GPU market[6] but, is the demand for accelerated 3D rendering still as prevalent or do video games impact the GPU market because video games are simply popular? This analysis looks for correlation between GPU sales and video games.

# Names

- Cameron VanderTuig
- Erdogan Ergit
- Henry Chan
- Wilson Tan

<a id='research_question'></a>
# Research Question

<font size="+1"> How does the release of popular and graphically intensive games correlate with GPU sales?</font>

<a id='background'></a>

## Background & Prior Work

The graphics processing unit (GPU) and the central processing unit (CPU) are perhaps the two of the most important parts of PC gaming. "A GPU is a single-chip processor that’s used chiefly to manage and enhance video and graphics performance."[1] A key difference between the two is the GPU's ability to parallel process which gives accelerated 3D rendering times.[3] Therefore, powerful games usually require a powerful GPU. Video games have also been getting more advanced in terms of their video quality and AI.[2] They're gradually becoming more computationally intensive as graphics continue to improve.[5] As video game graphics and GPU performance grows, it's unclear how the relation between video game graphics and the GPU market has grown. We know that video games still have a massive impact on the GPU market today since "The gaming GPU market is expected to grow at 14.1% CAGR during the forecast period of 2021-2026."[6] However, there has been a lack of research and understanding, beyond the traditional perspective, about how the new releases drive hardware purchases. So, we aim to determine if the new game releases affect computer hardware sales. 

References (include links):
- 1)  https://www.hp.com/us-en/shop/tech-takes/gpu-vs-cpu-for-pc-gaming
- 2)  https://www.thelogocreative.co.uk/the-evolution-of-video-game-graphics/ 
- 3)  https://medium.com/analytics-vidhya/gpu-for-deep-learning-7f4ef099b702
- 4)  https://www.investopedia.com/terms/g/graphics-processing-unit-gpu.asp
- 5)  https://www.intel.com/content/www/us/en/products/docs/processors/what-is-a-gpu.html
- 6)  https://www.mordorintelligence.com/industry-reports/gaming-gpu-market
- 7)  https://www.statista.com/statistics/552623/number-games-released-steam/

# Hypothesis


We predict that the release of popular, graphically intensive video games is positively correlated with computer hardware sales. Traditionally, computer hardware sales would increase with the arrival of new popular games. Most of the time, new games would come with better graphics, which would require better hardware. So, in this case, one might predict that computer hardware sales might increase with the arrival of new popular games.

# Dataset(s)

- Dataset Name: Steam Hardware & Software Survey
- Link to the dataset: https://web.archive.org/web/yyyymmddhhmmss/http://store.steampowered.com/hwsurvey/videocard
- yyyymmddhhmmss: date of the range in 2017 to 2020
- Number of observations: 2162
- This dataset is the survey that steam users can opt into to measure which hardware they have.
---
- Dataset Name: Benchmarking Data
- Link to the dataset: https://benchmarks.ul.com/hardware/gpu/
- Number of observations: 110
- Contains the relative performance statistics for many graphics hardware systems. It gives us a way to equate a GPU to the performance it is capable of.
---
- Dataset Name: Game_Main
- Link to the dataset: https://www.kaggle.com/deepann/80000-steam-games-dataset/version/1
- Number of observations: 81048
- This dataset is a compiled set of scrapable steam game data. This particular set contains review statistics, game names, release dates, and urls to the steam page.
---
Ultimately, we want to know how releases of videogames affect sales of computer hardware, so we need an avenue to get from video games and their popularity to computer hardware sales.  
The way we anticipate doing that now is like so:  
Video game popularity and release date (Game_Main.csv) -->   
Video game's hardware requirements (Game_Requirements.csv) -->   
Hardware in those requirements (Game_Requirements.csv) -->   
Level of graphical intensity (Benchmarking_Data.csv) -->   
Hardware that can handle that intensity (Benchmarking_Data.csv) -->   
Sales and release date of hardware (Steam Hardware & Software_Survey.csv).

# Setup


```python
# run only if you have not install waybackpy
#pip install waybackpy
```


```python
# run only if you have not install aiohttp
#pip install aiohttp
```


```python
import aiohttp
import asyncio
import requests
import patsy
import os
import re
# import time
# import timeit
import waybackpy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from bs4 import BeautifulSoup
from dateutil.parser import parse
from statsmodels.stats.outliers_influence import variance_inflation_factor
```


```python
# Helper function to get webpages content
async def fetch(session, url):
    async with session.get(url) as response:
        if response.status != 200:
            response.raise_for_status()
        return await response.text()
```


```python
async def fetch_all(session, urls):
    tasks = []
    for url in urls:
        task = asyncio.create_task(fetch(session, url))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results
```


```python
start_year = '2017'
end_year = '2020'
dates = pd.date_range(start=start_year, end=end_year, freq='M', closed=None) + pd.DateOffset(days=15)

dates_times = dates.strftime('%Y%m%d%H%M%S')
dates_months = dates.strftime('%h_%Y')
```


```python
url = 'https://store.steampowered.com/hwsurvey/videocard'
wayback_url = 'https://web.archive.org/web/'
wayback = waybackpy.Url(url)

steam_url = [wayback_url + date + '/' + url for date in dates_times]
```


```python
# Read wayback machine webpage
connector = aiohttp.TCPConnector(limit=10)
async with aiohttp.ClientSession(connector=connector) as session:
    pages = await fetch_all(session, steam_url)
connector.close()
```




    <aiohttp.connector._DeprecationWaiter at 0x1a1015a9250>



#### Web Scraping: GPUs

Using the code below, we scraped for GPU data per month in a span of 2017-2019 from Steam and generated CSV files per month within the time span. This time span was chosen because months in 2016 didn't contain DirectX 12 GPUs and there existed a chip shortage for years after 2019 (2020, 2021).  
NOTE: The code below will take a bit of time (5-10 minutes).

The data we get from this is the name and change in percentage of usage per month. The name will be used later to determine if a game was graphically demanding for the time it was released in. Since we were unable to find hard sales number of GPUs, we will use the change of usage of GPUs as an estimation to sales number.


```python
# Set the output directory
outdir = './GPU_Data'
if not os.path.exists(outdir):
    os.mkdir(outdir)
```


```python
# Get Content as .csv file
for i in range(len(dates)):
    
    soup = BeautifulSoup(pages[i], 'html.parser')
    
    # Get names and increase/decrease percentages
    name = soup.find_all('div', {'class', 'substats_col_left'})
    stat = soup.find_all('span', {'class', 'stat_increase', 'stat_decrease', 'stat_unchanged'})

    names = []
    stats = []
    performances = []
    directx_version = []

    for tag in stat:
        stats.append(tag.text.strip())

    for _ in range(1, 7):
        stats.pop(0)

    temp = ''
    unwanted = ['overall distribution of cards', 'directx 11 gpus', 'directx 10 gpus', 'directx 9 shader model 2b and 3.0 gpus',
                'directx 9 shader model 2.0 gpus', 'directx 8 gpus and below', 'all video cards']

    # Directx 12 gpus
    directx = 0

    # Append only the wanted names
    for tag in name:
        temp = tag.text.strip()
        if (temp.lower() in unwanted):
            directx = 0
            continue

        # Limit to 3 most recent DirectX versions 
        if (temp.lower() == 'directx 12 gpus'):
            directx = 12
            continue

        directx_version.append(directx)
        names.append(temp)

    df = pd.DataFrame({'GPU Name': names, 'Change (%)': stats, 'DirectX Version': directx_version})
    df = df[df['DirectX Version'] != 0]
    df.set_index(np.array(range(0, df.shape[0])), inplace=True)
    df.to_csv('./GPU_Data/' + dates_months[i] + '_GPUs.csv', encoding='utf-8', index=False)
```

From the code above, we get multiple CSV files. These files are contained in the list "files". Since every month will have duplicate GPU names, we can create a set, "GPUs", to get unique GPUs only.


```python
GPUs = set([])
for date in dates_months:
    df = pd.read_csv('./GPU_Data/' + date + '_GPUs.csv')
    df = df[df['GPU Name'] != 'Other']
    df['GPU Name'].apply(lambda x: GPUs.add(x))
    
GPUs
```




    {'AMD Radeon HD 7480D',
     'AMD Radeon HD 7540D',
     'AMD Radeon HD 7700 Series',
     'AMD Radeon HD 7800 Series',
     'AMD Radeon HD 7900 Series',
     'AMD Radeon HD 8370D',
     'AMD Radeon HD 8470D',
     'AMD Radeon HD 8500 Series',
     'AMD Radeon HD 8600 Series',
     'AMD Radeon HD 8800 Series',
     'AMD Radeon R4 Graphics',
     'AMD Radeon R5 Graphics',
     'AMD Radeon R5 M330',
     'AMD Radeon R7 200 Series',
     'AMD Radeon R7 300 Series',
     'AMD Radeon R7 Graphics',
     'AMD Radeon R9 200 Series',
     'AMD Radeon R9 380 Series',
     'AMD Radeon R9 390 Series',
     'AMD Radeon RX 460',
     'AMD Radeon RX 470',
     'AMD Radeon RX 480',
     'AMD Radeon RX 550',
     'AMD Radeon RX 560',
     'AMD Radeon RX 570',
     'AMD Radeon RX 580',
     'AMD Radeon RX Vega',
     'AMD Radeon Vega 8 Graphics',
     'Intel HD Graphics 4400',
     'Intel HD Graphics 4600',
     'Intel HD Graphics 5000',
     'Intel HD Graphics 520',
     'Intel HD Graphics 530',
     'Intel HD Graphics 5500',
     'Intel HD Graphics 6000',
     'Intel HD Graphics 620',
     'Intel HD Graphics 630',
     'Intel Iris Graphics 5100',
     'Intel Iris Graphics 6100',
     'Intel Iris Plus Graphics 640',
     'Intel Iris Pro Graphics 5200',
     'Intel UHD Graphics 620',
     'Intel UHD Graphics 630',
     'NVIDIA GeForce 610M',
     'NVIDIA GeForce 840M',
     'NVIDIA GeForce 920M',
     'NVIDIA GeForce 920MX',
     'NVIDIA GeForce 940M',
     'NVIDIA GeForce 940MX',
     'NVIDIA GeForce GT 1030',
     'NVIDIA GeForce GT 430',
     'NVIDIA GeForce GT 440',
     'NVIDIA GeForce GT 540M',
     'NVIDIA GeForce GT 610',
     'NVIDIA GeForce GT 630',
     'NVIDIA GeForce GT 630M',
     'NVIDIA GeForce GT 640',
     'NVIDIA GeForce GT 650M',
     'NVIDIA GeForce GT 710',
     'NVIDIA GeForce GT 720M',
     'NVIDIA GeForce GT 730',
     'NVIDIA GeForce GT 740',
     'NVIDIA GeForce GT 740M',
     'NVIDIA GeForce GT 750M',
     'NVIDIA GeForce GTS 450',
     'NVIDIA GeForce GTX 1050',
     'NVIDIA GeForce GTX 1050 Ti',
     'NVIDIA GeForce GTX 1060',
     'NVIDIA GeForce GTX 1070',
     'NVIDIA GeForce GTX 1070 Ti',
     'NVIDIA GeForce GTX 1080',
     'NVIDIA GeForce GTX 1080 Ti',
     'NVIDIA GeForce GTX 1650',
     'NVIDIA GeForce GTX 1660',
     'NVIDIA GeForce GTX 1660 Ti',
     'NVIDIA GeForce GTX 550 Ti',
     'NVIDIA GeForce GTX 560',
     'NVIDIA GeForce GTX 560 Ti',
     'NVIDIA GeForce GTX 650',
     'NVIDIA GeForce GTX 650 Ti',
     'NVIDIA GeForce GTX 660',
     'NVIDIA GeForce GTX 660 Ti',
     'NVIDIA GeForce GTX 670',
     'NVIDIA GeForce GTX 680',
     'NVIDIA GeForce GTX 745',
     'NVIDIA GeForce GTX 750',
     'NVIDIA GeForce GTX 750 Ti',
     'NVIDIA GeForce GTX 760',
     'NVIDIA GeForce GTX 770',
     'NVIDIA GeForce GTX 780',
     'NVIDIA GeForce GTX 850M',
     'NVIDIA GeForce GTX 860M',
     'NVIDIA GeForce GTX 950',
     'NVIDIA GeForce GTX 950M',
     'NVIDIA GeForce GTX 960',
     'NVIDIA GeForce GTX 960M',
     'NVIDIA GeForce GTX 965M',
     'NVIDIA GeForce GTX 970',
     'NVIDIA GeForce GTX 970M',
     'NVIDIA GeForce GTX 980',
     'NVIDIA GeForce GTX 980 Ti',
     'NVIDIA GeForce GTX 980M',
     'NVIDIA GeForce MX150',
     'NVIDIA GeForce RTX 2060',
     'NVIDIA GeForce RTX 2060 SUPER',
     'NVIDIA GeForce RTX 2070',
     'NVIDIA GeForce RTX 2070 SUPER',
     'NVIDIA GeForce RTX 2080',
     'NVIDIA GeForce RTX 2080 SUPER',
     'NVIDIA GeForce RTX 2080 Ti'}




```python
GPUs = list(GPUs)
```

#### Web Scraping: Performance

Data from Steam only provided us with the name and change in usage percentage. However, in order to determine the performance of a GPU, we would need to scrape a different site, "benchmarks.ul.com".

Performance will tell us whether a game is demanding for when it was released, and how new/old a GPU is. Newer GPUs have higher performances, so when given a list of GPUs and their performances, we're able to determine which GPU are newest.


```python
connector = aiohttp.TCPConnector()
ul_url = ('https://benchmarks.ul.com/hardware/gpu/' + gpu + ' review' for gpu in GPUs)
async with aiohttp.ClientSession(connector=connector) as session:
    ul_ratings = await fetch_all(session, ul_url)
connector.close()
```




    <aiohttp.connector._DeprecationWaiter at 0x1a1029da9a0>




```python
performances = []
for rating in ul_ratings:
    soup = BeautifulSoup(rating, 'html.parser')

    try:
        performance = soup.find_all('span', {'class', 'result-pimp-badge-score-item'})[0].text.strip()
        performances.append(performance)
    except:
        performances.append(0)
```

With the list of GPUs and performances, we can create a dictionary to easily grab the performance of the GPU


```python
GPU_dict = dict(zip(GPUs, performances))
GPU_dict
```




    {'AMD Radeon R7 300 Series': 0,
     'NVIDIA GeForce RTX 2060': '7592',
     'AMD Radeon HD 7900 Series': 0,
     'AMD Radeon RX 550': '1193',
     'AMD Radeon RX Vega': 0,
     'NVIDIA GeForce 940M': '514',
     'Intel UHD Graphics 630': '449',
     'AMD Radeon R7 Graphics': 0,
     'AMD Radeon RX 470': '3643',
     'NVIDIA GeForce GTX 560': 0,
     'NVIDIA GeForce GTX 980 Ti': '5802',
     'NVIDIA GeForce GTX 750 Ti': '1285',
     'AMD Radeon HD 7700 Series': 0,
     'AMD Radeon HD 8370D': 0,
     'AMD Radeon HD 7480D': 0,
     'NVIDIA GeForce 610M': 0,
     'NVIDIA GeForce RTX 2070': '9125',
     'NVIDIA GeForce RTX 2070 SUPER': '10171',
     'AMD Radeon R9 200 Series': 0,
     'NVIDIA GeForce RTX 2080 Ti': '14761',
     'Intel HD Graphics 630': '408',
     'NVIDIA GeForce GTX 1070': '6083',
     'NVIDIA GeForce GTX 1070 Ti': '6825',
     'Intel HD Graphics 530': '381',
     'NVIDIA GeForce GTX 660': '1323',
     'NVIDIA GeForce GTX 560 Ti': '389',
     'NVIDIA GeForce RTX 2080 SUPER': '11670',
     'NVIDIA GeForce 940MX': '599',
     'NVIDIA GeForce GTX 1660': '5469',
     'NVIDIA GeForce GTX 965M': '1809',
     'AMD Radeon Vega 8 Graphics': 0,
     'AMD Radeon R5 Graphics': 0,
     'NVIDIA GeForce GT 430': 0,
     'NVIDIA GeForce GTX 860M': '1156',
     'AMD Radeon R5 M330': 0,
     'Intel Iris Plus Graphics 640': 0,
     'AMD Radeon HD 8600 Series': 0,
     'NVIDIA GeForce GTX 1660 Ti': '6379',
     'NVIDIA GeForce GTX 950M': 0,
     'NVIDIA GeForce RTX 2080': '11091',
     'NVIDIA GeForce GTX 680': '2019',
     'NVIDIA GeForce GTX 1080': '7584',
     'AMD Radeon R4 Graphics': 0,
     'AMD Radeon RX 580': '4294',
     'NVIDIA GeForce GT 640': 0,
     'NVIDIA GeForce GT 650M': '414',
     'NVIDIA GeForce GTX 850M': '983',
     'Intel HD Graphics 5500': '234',
     'AMD Radeon HD 8470D': 0,
     'AMD Radeon HD 7800 Series': 0,
     'NVIDIA GeForce GTX 745': '656',
     'AMD Radeon HD 8500 Series': 0,
     'NVIDIA GeForce GTX 780': '2844',
     'Intel HD Graphics 5000': 0,
     'Intel Iris Graphics 5100': 0,
     'NVIDIA GeForce 840M': '500',
     'NVIDIA GeForce GTX 1650': '3649',
     'Intel HD Graphics 520': 0,
     'NVIDIA GeForce GTX 970M': '2282',
     'Intel UHD Graphics 620': '355',
     'NVIDIA GeForce GT 740': '513',
     'AMD Radeon RX 480': '4087',
     'NVIDIA GeForce GT 630': '166',
     'Intel Iris Graphics 6100': 0,
     'NVIDIA GeForce RTX 2060 SUPER': '8813',
     'NVIDIA GeForce GTX 670': '1856',
     'NVIDIA GeForce GTX 970': '3659',
     'AMD Radeon R9 380 Series': 0,
     'NVIDIA GeForce GTX 760': '1690',
     'Intel HD Graphics 6000': 0,
     'NVIDIA GeForce GT 610': '88',
     'NVIDIA GeForce GTS 450': 0,
     'NVIDIA GeForce GT 540M': 0,
     'NVIDIA GeForce GTX 660 Ti': '1639',
     'AMD Radeon RX 570': '3812',
     'NVIDIA GeForce GTX 1080 Ti': '9926',
     'Intel HD Graphics 4600': '194',
     'NVIDIA GeForce GTX 650': '545',
     'NVIDIA GeForce GTX 750': '1058',
     'NVIDIA GeForce GT 440': 0,
     'NVIDIA GeForce GTX 1050': '1738',
     'AMD Radeon R9 390 Series': 0,
     'NVIDIA GeForce GT 630M': 0,
     'NVIDIA GeForce GTX 980M': '2932',
     'NVIDIA GeForce GT 750M': '461',
     'NVIDIA GeForce GTX 550 Ti': '322',
     'NVIDIA GeForce GTX 960M': '1240',
     'NVIDIA GeForce MX150': '996',
     'NVIDIA GeForce GTX 770': '2158',
     'NVIDIA GeForce 920M': '326',
     'NVIDIA GeForce 920MX': '384',
     'Intel HD Graphics 4400': 0,
     'NVIDIA GeForce GTX 650 Ti': '906',
     'AMD Radeon HD 7540D': 0,
     'NVIDIA GeForce GT 730': '299',
     'AMD Radeon RX 460': '1739',
     'NVIDIA GeForce GT 710': '200',
     'NVIDIA GeForce GT 720M': 0,
     'NVIDIA GeForce GTX 950': '1922',
     'AMD Radeon RX 560': '1840',
     'NVIDIA GeForce GTX 1050 Ti': '2356',
     'Intel Iris Pro Graphics 5200': 0,
     'NVIDIA GeForce GTX 960': '2308',
     'NVIDIA GeForce GTX 980': '4383',
     'Intel HD Graphics 620': '335',
     'NVIDIA GeForce GT 1030': '1091',
     'AMD Radeon R7 200 Series': 0,
     'NVIDIA GeForce GT 740M': '345',
     'NVIDIA GeForce GTX 1060': '3740',
     'AMD Radeon HD 8800 Series': 0}



Now that we have GPU names, percentage change per month, and performances, we have complete web scraping for GPUs.

#### Games

With GPU names and their performances, we're able to determine whether the GPU is new/old and whether a game is demanding or not. However, we would also need a list of games within our time frame to measure whether release of popular, demanding games affect the sales number of GPUs.

Because we're using Steam data, we will only be taking a look at games on Steam.

By using the Steam reviews of the games, we can determine the popularity of the game. URLs and names will be used to simply indicate what game we're looking at and date will be their release date


```python
# Create dataframe (Game name)
df_game_main = pd.read_csv('./Data/Game_Main.csv')
df_game_main
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>name</th>
      <th>all_reviews</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://store.steampowered.com/app/945360/Amon...</td>
      <td>Among Us</td>
      <td>Overwhelmingly Positive(224,878)- 95% of the 2...</td>
      <td>16-Nov-18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://store.steampowered.com/app/730/Counter...</td>
      <td>Counter-Strike: Global Offensive</td>
      <td>Very Positive(4,843,904)- 87% of the 4,843,904...</td>
      <td>21-Aug-12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://store.steampowered.com/app/1097150/Fal...</td>
      <td>Fall Guys: Ultimate Knockout</td>
      <td>Very Positive(223,706)- 80% of the 223,706 use...</td>
      <td>3-Aug-20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://store.steampowered.com/app/1158310/Cru...</td>
      <td>Crusader Kings III</td>
      <td>Very Positive(18,951)- 92% of the 18,951 user ...</td>
      <td>1-Sep-20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://store.steampowered.com/app/1085660/Des...</td>
      <td>Destiny 2</td>
      <td>Very Positive(284,689)- 86% of the 284,689 use...</td>
      <td>1-Oct-19</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>81043</th>
      <td>https://store.steampowered.com/bundle/2961/Ste...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>81044</th>
      <td>https://store.steampowered.com/bundle/3123/Det...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>81045</th>
      <td>https://store.steampowered.com/bundle/3175/Fea...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>81046</th>
      <td>https://store.steampowered.com/bundle/3176/Fea...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <th>81047</th>
      <td>https://store.steampowered.com/bundle/3237/Sho...</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
<p>81048 rows × 4 columns</p>
</div>



Requirements of a game will be needed to determine how demanding a game is.


```python
# Create dataframe (Game name)
df_game_requirements = pd.read_csv('./Data/Game_Requirements.csv')
df_game_requirements
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>requirements</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://store.steampowered.com/app/945360/Amon...</td>
      <td>System RequirementsMinimum:OS: Windows 7 SP1+P...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://store.steampowered.com/app/730/Counter...</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://store.steampowered.com/app/1097150/Fal...</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://store.steampowered.com/app/1158310/Cru...</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://store.steampowered.com/app/1085660/Des...</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>81043</th>
      <td>https://store.steampowered.com/bundle/2961/Ste...</td>
      <td>-</td>
    </tr>
    <tr>
      <th>81044</th>
      <td>https://store.steampowered.com/bundle/3123/Det...</td>
      <td>-</td>
    </tr>
    <tr>
      <th>81045</th>
      <td>https://store.steampowered.com/bundle/3175/Fea...</td>
      <td>-</td>
    </tr>
    <tr>
      <th>81046</th>
      <td>https://store.steampowered.com/bundle/3176/Fea...</td>
      <td>-</td>
    </tr>
    <tr>
      <th>81047</th>
      <td>https://store.steampowered.com/bundle/3237/Sho...</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
<p>81048 rows × 2 columns</p>
</div>



# Data Cleaning

#### Data Cleaning: GPUs

Although we scraped for the GPU data, we still have a few issues to clean up.

Remove Other and DirectX Version:
Data from Steam included an "Other" GPU label. This is unusable and needs to be dropped. During setup, we used the DirectX Version to pick out only DirectX Version 12 GPUs, but is no longer needed.

Get Performance list:
Earlier in our setup, we created a dictionary which used the GPU names as keys and performance as values. We use that here to grab the performance values for the GPUs in a month.

Replace 0 values:
When scraping for performances, some values came back as 0 which indicated that it didn't have a performance score. We can fill in these 0 values by using the mean performance of the GPUs for that month.
NOTE: The mean performance score was intentionally left as an integer. Performance data scraped are either originally integers, or were converted to integers. Also, the performance score will be used to determine the higher performing GPUs that month and keeping it as an integer will have no effect.

Get GPUs above 75th percentile:
To determine which GPUs are high performing, we are taking everything above the 75th percentile.


```python
for x in dates_months:
    
    # Remove Other and DirectX Version
    performances = []
    df = pd.read_csv('./GPU_Data/' + x + '_GPUs.csv')
    df = df[df['GPU Name'] != 'Other']
    df.drop(labels='DirectX Version', axis=1, inplace=True)
    
    # Get performance list
    performances = list(df['GPU Name'].apply(lambda x: GPU_dict.get(x)))
    
    df['Performance (3DMark)'] = performances
    df['Performance (3DMark)'] = df['Performance (3DMark)'].apply(lambda x: int(x))
    
    # Replace 0 values. Took mean as an integer because Performance scores were also converted to ints
    mean = int(df['Performance (3DMark)'].mean())
    df['Performance (3DMark)'] = df['Performance (3DMark)'].replace(0, mean)
    
    # Get GPUs above 75th percentile
    percentile = df['Performance (3DMark)'].quantile([0.75]).loc[0.75]
    df = df.loc[df['Performance (3DMark)'] > percentile]
    df = df.set_index(np.array(range(0, df.shape[0])))
    df.to_csv('./GPU_Data/' + x + '_75_ptile.csv', encoding='utf-8', index=False)
```

#### Data Cleaning: Games

We can merge df_game_main and df_game_requirements


```python
df_game = pd.merge(df_game_main, df_game_requirements)
```

We then drop any empty row (when it equal to '-')


```python
df_game.replace('-', np.nan, inplace=True)
df_game.dropna(inplace=True)

df_game
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>name</th>
      <th>all_reviews</th>
      <th>date</th>
      <th>requirements</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://store.steampowered.com/app/945360/Amon...</td>
      <td>Among Us</td>
      <td>Overwhelmingly Positive(224,878)- 95% of the 2...</td>
      <td>16-Nov-18</td>
      <td>System RequirementsMinimum:OS: Windows 7 SP1+P...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://store.steampowered.com/app/945360/Amon...</td>
      <td>Among Us</td>
      <td>Overwhelmingly Positive(224,878)- 95% of the 2...</td>
      <td>16-Nov-18</td>
      <td>System RequirementsMinimum:OS: Windows 7 SP1+P...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://store.steampowered.com/app/945360/Amon...</td>
      <td>Among Us</td>
      <td>Overwhelmingly Positive(224,878)- 95% of the 2...</td>
      <td>16-Nov-18</td>
      <td>System RequirementsMinimum:OS: Windows 7 SP1+P...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://store.steampowered.com/app/945360/Amon...</td>
      <td>Among Us</td>
      <td>Overwhelmingly Positive(224,878)- 95% of the 2...</td>
      <td>16-Nov-18</td>
      <td>System RequirementsMinimum:OS: Windows 7 SP1+P...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://store.steampowered.com/app/945360/Amon...</td>
      <td>Among Us</td>
      <td>Overwhelmingly Positive(224,878)- 95% of the 2...</td>
      <td>16-Nov-18</td>
      <td>System RequirementsMinimum:OS: Windows 7 SP1+P...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>81379</th>
      <td>https://store.steampowered.com/app/1346541/STA...</td>
      <td>STAR WARS™: The Old Republic™ - Sith Bundles</td>
      <td>21 Jul, 2020</td>
      <td>21 Jul, 2020</td>
      <td>System RequirementsMinimum:OS: Windows 7 or la...</td>
    </tr>
    <tr>
      <th>81380</th>
      <td>https://store.steampowered.com/app/1347970/Pat...</td>
      <td>Patch Quest</td>
      <td>Late 2020</td>
      <td>Late 2020</td>
      <td>System RequirementsMinimum:OS: Windows 10Proce...</td>
    </tr>
    <tr>
      <th>81381</th>
      <td>https://store.steampowered.com/app/1349120/_/?...</td>
      <td>球球少女</td>
      <td>6 Nov, 2020</td>
      <td>6 Nov, 2020</td>
      <td>System RequirementsMinimum:OS: WIN7 SP1/WIN8/W...</td>
    </tr>
    <tr>
      <th>81382</th>
      <td>https://store.steampowered.com/app/1349170/Fur...</td>
      <td>Furries &amp; Scalies &amp; Bears OH MY! 2: Return to ...</td>
      <td>20 Apr, 2021</td>
      <td>20 Apr, 2021</td>
      <td>System RequirementsWindowsSteamOS + LinuxMinim...</td>
    </tr>
    <tr>
      <th>81402</th>
      <td>https://store.steampowered.com/app/597700/OVIV...</td>
      <td>OVIVO</td>
      <td>12 May, 2017</td>
      <td>12 May, 2017</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
    </tr>
  </tbody>
</table>
<p>74725 rows × 5 columns</p>
</div>



Remove duplicates


```python
df_game.drop_duplicates(subset='name', keep='first', inplace=True)
df_game.reset_index(drop=True, inplace=True)

df_game
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>name</th>
      <th>all_reviews</th>
      <th>date</th>
      <th>requirements</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://store.steampowered.com/app/945360/Amon...</td>
      <td>Among Us</td>
      <td>Overwhelmingly Positive(224,878)- 95% of the 2...</td>
      <td>16-Nov-18</td>
      <td>System RequirementsMinimum:OS: Windows 7 SP1+P...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://store.steampowered.com/app/730/Counter...</td>
      <td>Counter-Strike: Global Offensive</td>
      <td>Very Positive(4,843,904)- 87% of the 4,843,904...</td>
      <td>21-Aug-12</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://store.steampowered.com/app/1097150/Fal...</td>
      <td>Fall Guys: Ultimate Knockout</td>
      <td>Very Positive(223,706)- 80% of the 223,706 use...</td>
      <td>3-Aug-20</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://store.steampowered.com/app/1158310/Cru...</td>
      <td>Crusader Kings III</td>
      <td>Very Positive(18,951)- 92% of the 18,951 user ...</td>
      <td>1-Sep-20</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://store.steampowered.com/app/1085660/Des...</td>
      <td>Destiny 2</td>
      <td>Very Positive(284,689)- 86% of the 284,689 use...</td>
      <td>1-Oct-19</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>63460</th>
      <td>https://store.steampowered.com/app/1340160/RIO...</td>
      <td>RIO - Raised In Oblivion</td>
      <td>30 Oct, 2020</td>
      <td>30 Oct, 2020</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
    </tr>
    <tr>
      <th>63461</th>
      <td>https://store.steampowered.com/app/1340360/Pol...</td>
      <td>Polycalypse: Last bit of Hope</td>
      <td>2020</td>
      <td>2020</td>
      <td>System RequirementsMinimum:OS: Windows 7 or la...</td>
    </tr>
    <tr>
      <th>63462</th>
      <td>https://store.steampowered.com/app/1341230/Sup...</td>
      <td>Super Buckyball Tournament Preseason</td>
      <td>Coming Soon</td>
      <td>Coming Soon</td>
      <td>System RequirementsMinimum:OS: Windows 7+ / 8....</td>
    </tr>
    <tr>
      <th>63463</th>
      <td>https://store.steampowered.com/app/1342050/Lis...</td>
      <td>Lisa and the Grimoire</td>
      <td>TBA</td>
      <td>TBA</td>
      <td>System RequirementsMinimum:OS: Windows® 7/8/8....</td>
    </tr>
    <tr>
      <th>63464</th>
      <td>https://store.steampowered.com/app/1343340/HOR...</td>
      <td>HORROR TALES: The Beggar</td>
      <td>1 Dec, 2020</td>
      <td>1 Dec, 2020</td>
      <td>System RequirementsMinimum:OS: Windows 7 SP1 6...</td>
    </tr>
  </tbody>
</table>
<p>63465 rows × 5 columns</p>
</div>



We're only interested in number of reviews and percentage of likes. We can use regular expressions and the apply function to get rid of useless information.


```python
df_game['all_reviews'] = df_game['all_reviews'].apply(lambda x: re.sub('[^0-9\s]', '', x))

df_game['all_reviews']
```




    0           224878 95   224878       
    1         4843904 87   4843904       
    2           223706 80   223706       
    3             18951 92   18951       
    4           284689 86   284689       
                         ...             
    63460                        30  2020
    63461                            2020
    63462                                
    63463                                
    63464                         1  2020
    Name: all_reviews, Length: 63465, dtype: object



Notice how the bottom 5 reviews are different from the top 5. Some reviews are unusable to us, so we can get rid of those rows.


```python
df_game = df_game[df_game['all_reviews'].str.split().apply(len) == 3]
df_game.reset_index(drop=True, inplace=True)

df_game
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>name</th>
      <th>all_reviews</th>
      <th>date</th>
      <th>requirements</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://store.steampowered.com/app/945360/Amon...</td>
      <td>Among Us</td>
      <td>224878 95   224878</td>
      <td>16-Nov-18</td>
      <td>System RequirementsMinimum:OS: Windows 7 SP1+P...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://store.steampowered.com/app/730/Counter...</td>
      <td>Counter-Strike: Global Offensive</td>
      <td>4843904 87   4843904</td>
      <td>21-Aug-12</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://store.steampowered.com/app/1097150/Fal...</td>
      <td>Fall Guys: Ultimate Knockout</td>
      <td>223706 80   223706</td>
      <td>3-Aug-20</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://store.steampowered.com/app/1158310/Cru...</td>
      <td>Crusader Kings III</td>
      <td>18951 92   18951</td>
      <td>1-Sep-20</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://store.steampowered.com/app/1085660/Des...</td>
      <td>Destiny 2</td>
      <td>284689 86   284689</td>
      <td>1-Oct-19</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4227</th>
      <td>https://store.steampowered.com/app/1236180/The...</td>
      <td>The Whipper</td>
      <td>4 20211 2022</td>
      <td>Q4 2021/Q1 2022</td>
      <td>System RequirementsWindowsSteamOS + LinuxMinim...</td>
    </tr>
    <tr>
      <th>4228</th>
      <td>https://store.steampowered.com/app/1361980/Wel...</td>
      <td>Welcome To... Chichester 2 - Part III : NightFall</td>
      <td>3  4 2021</td>
      <td>Q3 - Q4 2021</td>
      <td>System RequirementsWindowsSteamOS + LinuxMinim...</td>
    </tr>
    <tr>
      <th>4229</th>
      <td>https://store.steampowered.com/app/1333910/Siz...</td>
      <td>Sizeable</td>
      <td>3  4 2020</td>
      <td>Coming Q3 - Q4 2020</td>
      <td>System RequirementsMinimum:OS: Windows 7 or ne...</td>
    </tr>
    <tr>
      <th>4230</th>
      <td>https://store.steampowered.com/app/346290/Penu...</td>
      <td>Penumbra: Necrologue</td>
      <td>508 88   508</td>
      <td>6 Feb, 2015</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
    </tr>
    <tr>
      <th>4231</th>
      <td>https://store.steampowered.com/app/1082450/Gol...</td>
      <td>Gold Hunter</td>
      <td>4 2020   2022</td>
      <td>public demo Q4 2020 [Release Steam 2022]</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
    </tr>
  </tbody>
</table>
<p>4232 rows × 5 columns</p>
</div>



With the useless information gone, we can get the number of reviews and percetange of likes.


```python
num_reviews = []
pct_like = []

for x in range(0, df_game.shape[0]):
    review = df_game.loc[x, 'all_reviews'].split()
    
    num_reviews.append(int(review[0]))
    pct_like.append(int(review[1]))
    
print(len(num_reviews), len(pct_like))
```

    4232 4232
    

Adding number of reviews, percentage of likes, and number of positive review columns. Also getting rid of the "all_reviews" column.


```python
num_pos_reviews = []
df_game = df_game.assign(num_reviews=num_reviews, pct_like=pct_like)

df_game['num_reviews'].apply(lambda x: int(x))
df_game['pct_like'].apply(lambda x: int(x))

for x in range(0, len(num_reviews)):
    pos_reviews = (float(num_reviews[x]) * (float(pct_like[x])/100))
    num_pos_reviews.append(int(pos_reviews))

df_game = df_game.assign(num_positive_reviews=num_pos_reviews)
df_game.drop(columns='all_reviews', inplace=True)
df_game
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>name</th>
      <th>date</th>
      <th>requirements</th>
      <th>num_reviews</th>
      <th>pct_like</th>
      <th>num_positive_reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://store.steampowered.com/app/945360/Amon...</td>
      <td>Among Us</td>
      <td>16-Nov-18</td>
      <td>System RequirementsMinimum:OS: Windows 7 SP1+P...</td>
      <td>224878</td>
      <td>95</td>
      <td>213634</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://store.steampowered.com/app/730/Counter...</td>
      <td>Counter-Strike: Global Offensive</td>
      <td>21-Aug-12</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
      <td>4843904</td>
      <td>87</td>
      <td>4214196</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://store.steampowered.com/app/1097150/Fal...</td>
      <td>Fall Guys: Ultimate Knockout</td>
      <td>3-Aug-20</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
      <td>223706</td>
      <td>80</td>
      <td>178964</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://store.steampowered.com/app/1158310/Cru...</td>
      <td>Crusader Kings III</td>
      <td>1-Sep-20</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
      <td>18951</td>
      <td>92</td>
      <td>17434</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://store.steampowered.com/app/1085660/Des...</td>
      <td>Destiny 2</td>
      <td>1-Oct-19</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
      <td>284689</td>
      <td>86</td>
      <td>244832</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4227</th>
      <td>https://store.steampowered.com/app/1236180/The...</td>
      <td>The Whipper</td>
      <td>Q4 2021/Q1 2022</td>
      <td>System RequirementsWindowsSteamOS + LinuxMinim...</td>
      <td>4</td>
      <td>20211</td>
      <td>808</td>
    </tr>
    <tr>
      <th>4228</th>
      <td>https://store.steampowered.com/app/1361980/Wel...</td>
      <td>Welcome To... Chichester 2 - Part III : NightFall</td>
      <td>Q3 - Q4 2021</td>
      <td>System RequirementsWindowsSteamOS + LinuxMinim...</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4229</th>
      <td>https://store.steampowered.com/app/1333910/Siz...</td>
      <td>Sizeable</td>
      <td>Coming Q3 - Q4 2020</td>
      <td>System RequirementsMinimum:OS: Windows 7 or ne...</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4230</th>
      <td>https://store.steampowered.com/app/346290/Penu...</td>
      <td>Penumbra: Necrologue</td>
      <td>6 Feb, 2015</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
      <td>508</td>
      <td>88</td>
      <td>447</td>
    </tr>
    <tr>
      <th>4231</th>
      <td>https://store.steampowered.com/app/1082450/Gol...</td>
      <td>Gold Hunter</td>
      <td>public demo Q4 2020 [Release Steam 2022]</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
      <td>4</td>
      <td>2020</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
<p>4232 rows × 7 columns</p>
</div>



#### Reformatting the date

Since the dates have different formats, we need a way to standardize them. In this case, we can write our own function to do this. Months will be rewritten as two digits numbers, years will be rewritten as four.

Our GPU data goes by months, so we can simply get rid of the days.

Because our time span of GPUs ranges from 2017-2019, we will only consider games within this time span.


```python
def standardize_date(date):
    try:
        return parse(date).strftime('%Y/%m')
    except:
        return np.nan
```


```python
df_game['date'] = df_game['date'].apply(standardize_date)
df_game.dropna(inplace=True)
df_game = df_game[(df_game['date'] >= '2017-01') & (df_game['date'] <= '2020-01')]
df_game = df_game.sort_values(by='num_reviews', ascending=False)
df_game.reset_index(drop=True, inplace=True)

df_game
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>name</th>
      <th>date</th>
      <th>requirements</th>
      <th>num_reviews</th>
      <th>pct_like</th>
      <th>num_positive_reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://store.steampowered.com/app/578080/PLAY...</td>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS</td>
      <td>2017/12</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
      <td>1316559</td>
      <td>52</td>
      <td>684610</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://store.steampowered.com/app/304930/Untu...</td>
      <td>Unturned</td>
      <td>2017/07</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
      <td>402298</td>
      <td>91</td>
      <td>366091</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://store.steampowered.com/app/252490/Rust...</td>
      <td>Rust</td>
      <td>2018/02</td>
      <td>System RequirementsWindowsMac OS XMinimum:Requ...</td>
      <td>375523</td>
      <td>84</td>
      <td>315439</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://store.steampowered.com/app/346110/ARK_...</td>
      <td>ARK: Survival Evolved</td>
      <td>2017/08</td>
      <td>System RequirementsWindowsMac OS XSteamOS + Li...</td>
      <td>289957</td>
      <td>78</td>
      <td>226166</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://store.steampowered.com/app/1085660/Des...</td>
      <td>Destiny 2</td>
      <td>2019/10</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
      <td>284689</td>
      <td>86</td>
      <td>244832</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1749</th>
      <td>https://store.steampowered.com/app/1172520/Col...</td>
      <td>Colorgrid</td>
      <td>2019/11</td>
      <td>System RequirementsWindowsMac OS XMinimum:OS: ...</td>
      <td>50</td>
      <td>100</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1750</th>
      <td>https://store.steampowered.com/app/1018130/Cas...</td>
      <td>Castle Break</td>
      <td>2019/02</td>
      <td>System RequirementsMinimum:OS: Windows 7 or ne...</td>
      <td>45</td>
      <td>95</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1751</th>
      <td>https://store.steampowered.com/app/763710/Rive...</td>
      <td>River City Melee Mach!!</td>
      <td>2019/10</td>
      <td>System RequirementsMinimum:Requires a 64-bit p...</td>
      <td>41</td>
      <td>85</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1752</th>
      <td>https://store.steampowered.com/app/705210/Cube...</td>
      <td>Cube Racer</td>
      <td>2017/10</td>
      <td>System RequirementsWindowsSteamOS + LinuxMinim...</td>
      <td>37</td>
      <td>89</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1753</th>
      <td>https://store.steampowered.com/app/523230/Supe...</td>
      <td>Super Spring Ninja</td>
      <td>2017/03</td>
      <td>System RequirementsMinimum:OS: 8Processor: Dua...</td>
      <td>16</td>
      <td>87</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
<p>1754 rows × 7 columns</p>
</div>



#### Web scraping for requirements

Some of the requirements are missing. However, since we have the urls for the games, we can easily web scrape for them instead.  
NOTE: This processes takes a while (2 - 3 minutes)


```python
connector = aiohttp.TCPConnector(limit=10)
cookies = {'birthtime': '283993201', 'mature_content': '1'}
async with aiohttp.ClientSession(connector=connector, cookies=cookies) as session:
    page = await fetch_all(session, df_game['url'])
connector.close()
```




    <aiohttp.connector._DeprecationWaiter at 0x1a106c35550>




```python
min_cards = []
rec_cards = []
minimum = ""
recommended = ""
contains_graphics_section = False

for index in range(0, df_game.shape[0]):
    try:
        soup = BeautifulSoup(page[index], 'html.parser')

        card = soup.find('div', {'class': 'game_area_sys_req sysreq_content active'})
        card = card.find_all('ul', {'class':'bb_ul'})

        for x in range(0, len(card)):
            contains_graphics_section = False
            rec = card[x].find_all('li')
            for tag in rec:
                if ('Graphics' in tag.text.strip()):
                    contains_graphics_section = True
                    if (x == 0):
                        minimum = tag.text.strip()
                        recommended = tag.text.strip()
                    elif (x == 1):
                        recommended = tag.text.strip()

        if (not contains_graphics_section):
            min_cards.append(np.nan)
            rec_cards.append(np.nan)
        else:
            min_cards.append(minimum)
            rec_cards.append(recommended)

    except:
        min_cards.append(np.nan)
        rec_cards.append(np.nan)
print(len(min_cards), len(rec_cards))
```

    1754 1754
    

Now we can add columns for minimum and recommended hardware.

Some of these games may not be on Steam anymore and the GPU requirements can no longer be scraped. In this case, we can just remove them from our dataset.


```python
df_game = df_game.assign(minimum=min_cards, recommended=rec_cards)
df_game = df_game.drop(labels='requirements', axis=1)
df_game.dropna(inplace=True)
df_game
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>name</th>
      <th>date</th>
      <th>num_reviews</th>
      <th>pct_like</th>
      <th>num_positive_reviews</th>
      <th>minimum</th>
      <th>recommended</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://store.steampowered.com/app/578080/PLAY...</td>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS</td>
      <td>2017/12</td>
      <td>1316559</td>
      <td>52</td>
      <td>684610</td>
      <td>Graphics: NVIDIA GeForce GTX 960 2GB / AMD Rad...</td>
      <td>Graphics: NVIDIA GeForce GTX 1060 3GB / AMD Ra...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://store.steampowered.com/app/252490/Rust...</td>
      <td>Rust</td>
      <td>2018/02</td>
      <td>375523</td>
      <td>84</td>
      <td>315439</td>
      <td>Graphics: GTX 670 2GB / AMD R9 280 better</td>
      <td>Graphics: GTX 980 / AMD R9 Fury</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://store.steampowered.com/app/346110/ARK_...</td>
      <td>ARK: Survival Evolved</td>
      <td>2017/08</td>
      <td>289957</td>
      <td>78</td>
      <td>226166</td>
      <td>Graphics: NVIDIA GTX 670 2GB/AMD Radeon HD 787...</td>
      <td>Graphics: NVIDIA GTX 670 2GB/AMD Radeon HD 787...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://store.steampowered.com/app/1085660/Des...</td>
      <td>Destiny 2</td>
      <td>2019/10</td>
      <td>284689</td>
      <td>86</td>
      <td>244832</td>
      <td>Graphics: NVIDIA® GeForce® GTX 660 2GB or GTX ...</td>
      <td>Graphics: NVIDIA® GeForce® GTX 970 4GB or GTX ...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>https://store.steampowered.com/app/444090/Pala...</td>
      <td>Paladins®</td>
      <td>2018/05</td>
      <td>280677</td>
      <td>85</td>
      <td>238575</td>
      <td>Graphics: Nvidia GeForce 8800 GT</td>
      <td>Graphics: Nvidia GeForce GTX 660 or ATI Radeon...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1741</th>
      <td>https://store.steampowered.com/app/930310/Puzz...</td>
      <td>Puzzle Plunder</td>
      <td>2018/10</td>
      <td>94</td>
      <td>93</td>
      <td>87</td>
      <td>Graphics: Videocard with at least 512MB</td>
      <td>Graphics: Videocard with at least 512MB</td>
    </tr>
    <tr>
      <th>1742</th>
      <td>https://store.steampowered.com/app/926520/Love...</td>
      <td>Love Letter</td>
      <td>2018/10</td>
      <td>92</td>
      <td>91</td>
      <td>83</td>
      <td>Graphics: Nvidia 450 GTS / Radeon HD 5750 or b...</td>
      <td>Graphics: Nvidia 650 GTS</td>
    </tr>
    <tr>
      <th>1749</th>
      <td>https://store.steampowered.com/app/1172520/Col...</td>
      <td>Colorgrid</td>
      <td>2019/11</td>
      <td>50</td>
      <td>100</td>
      <td>50</td>
      <td>Graphics: Graphics card supporting DirectX 9.0c</td>
      <td>Graphics: Graphics card supporting DirectX 9.0c</td>
    </tr>
    <tr>
      <th>1750</th>
      <td>https://store.steampowered.com/app/1018130/Cas...</td>
      <td>Castle Break</td>
      <td>2019/02</td>
      <td>45</td>
      <td>95</td>
      <td>42</td>
      <td>Graphics: existing</td>
      <td>Graphics: working with 1920x1080</td>
    </tr>
    <tr>
      <th>1752</th>
      <td>https://store.steampowered.com/app/705210/Cube...</td>
      <td>Cube Racer</td>
      <td>2017/10</td>
      <td>37</td>
      <td>89</td>
      <td>32</td>
      <td>Graphics: 512 MB</td>
      <td>Graphics: 1 GB</td>
    </tr>
  </tbody>
</table>
<p>1493 rows × 8 columns</p>
</div>



Remove "Graphics:" from the added columns


```python
df_game['minimum'] = df_game['minimum'].apply(lambda x: (x.replace('Graphics:', '')).strip())
df_game['recommended'] = df_game['recommended'].apply(lambda x: (x.replace('Graphics:', '')).strip())
```


```python
df_game.reset_index(drop=True, inplace=True)
df_game.isnull().any()
```




    url                     False
    name                    False
    date                    False
    num_reviews             False
    pct_like                False
    num_positive_reviews    False
    minimum                 False
    recommended             False
    dtype: bool



#### Popularity

For our measurement of popularity, we orignially had the number of reviews and the percentage of positive reviews. From these two values, we got the number of positive reviews. 

In our analysis, we can measure popularity in three ways. Number of reviews, percentage of positive reviews, and number of postive reviews.

Similar to how we determined whether a GPU is high performing, we can take the 75th percentile for each of these categories to determine if a game is considered popular in each of the number of reviews categories.

Taking the percentile of the percentage of likes for a game isn't an effective measurement since many games with a small number of reviews is volatile and can result in high numbers for percentage of likes. Instead, we will go by Steam's cutoff of 70%.


```python
percentile_revs = df_game['num_reviews'].quantile([0.75]).loc[0.75]
percentile_pos_revs = df_game['num_positive_reviews'].quantile([0.75]).loc[0.75]
percentile_pct = 70

popular_revs = [(x > percentile_revs) for x in df_game['num_reviews']]
popular_pct = [(x > percentile_pct) for x in df_game['pct_like']]
popular_pos_revs = [(x > percentile_pos_revs) for x in df_game['num_positive_reviews']]
```


```python
df_game = df_game.assign(popular_revs=popular_revs, popular_pos_revs=popular_pos_revs, popular_pct_likes=popular_pct)
df_game.replace([True, False], [1, 0], inplace=True)
```


```python
df_game
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>name</th>
      <th>date</th>
      <th>num_reviews</th>
      <th>pct_like</th>
      <th>num_positive_reviews</th>
      <th>minimum</th>
      <th>recommended</th>
      <th>popular_revs</th>
      <th>popular_pos_revs</th>
      <th>popular_pct_likes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://store.steampowered.com/app/578080/PLAY...</td>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS</td>
      <td>2017/12</td>
      <td>1316559</td>
      <td>52</td>
      <td>684610</td>
      <td>NVIDIA GeForce GTX 960 2GB / AMD Radeon R7 370...</td>
      <td>NVIDIA GeForce GTX 1060 3GB / AMD Radeon RX 58...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://store.steampowered.com/app/252490/Rust...</td>
      <td>Rust</td>
      <td>2018/02</td>
      <td>375523</td>
      <td>84</td>
      <td>315439</td>
      <td>GTX 670 2GB / AMD R9 280 better</td>
      <td>GTX 980 / AMD R9 Fury</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://store.steampowered.com/app/346110/ARK_...</td>
      <td>ARK: Survival Evolved</td>
      <td>2017/08</td>
      <td>289957</td>
      <td>78</td>
      <td>226166</td>
      <td>NVIDIA GTX 670 2GB/AMD Radeon HD 7870 2GB or b...</td>
      <td>NVIDIA GTX 670 2GB/AMD Radeon HD 7870 2GB or b...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://store.steampowered.com/app/1085660/Des...</td>
      <td>Destiny 2</td>
      <td>2019/10</td>
      <td>284689</td>
      <td>86</td>
      <td>244832</td>
      <td>NVIDIA® GeForce® GTX 660 2GB or GTX 1050 2GB /...</td>
      <td>NVIDIA® GeForce® GTX 970 4GB or GTX 1060 6GB /...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://store.steampowered.com/app/444090/Pala...</td>
      <td>Paladins®</td>
      <td>2018/05</td>
      <td>280677</td>
      <td>85</td>
      <td>238575</td>
      <td>Nvidia GeForce 8800 GT</td>
      <td>Nvidia GeForce GTX 660 or ATI Radeon HD 7950</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>https://store.steampowered.com/app/930310/Puzz...</td>
      <td>Puzzle Plunder</td>
      <td>2018/10</td>
      <td>94</td>
      <td>93</td>
      <td>87</td>
      <td>Videocard with at least 512MB</td>
      <td>Videocard with at least 512MB</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1489</th>
      <td>https://store.steampowered.com/app/926520/Love...</td>
      <td>Love Letter</td>
      <td>2018/10</td>
      <td>92</td>
      <td>91</td>
      <td>83</td>
      <td>Nvidia 450 GTS / Radeon HD 5750 or better</td>
      <td>Nvidia 650 GTS</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1490</th>
      <td>https://store.steampowered.com/app/1172520/Col...</td>
      <td>Colorgrid</td>
      <td>2019/11</td>
      <td>50</td>
      <td>100</td>
      <td>50</td>
      <td>Graphics card supporting DirectX 9.0c</td>
      <td>Graphics card supporting DirectX 9.0c</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1491</th>
      <td>https://store.steampowered.com/app/1018130/Cas...</td>
      <td>Castle Break</td>
      <td>2019/02</td>
      <td>45</td>
      <td>95</td>
      <td>42</td>
      <td>existing</td>
      <td>working with 1920x1080</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1492</th>
      <td>https://store.steampowered.com/app/705210/Cube...</td>
      <td>Cube Racer</td>
      <td>2017/10</td>
      <td>37</td>
      <td>89</td>
      <td>32</td>
      <td>512 MB</td>
      <td>1 GB</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1493 rows × 11 columns</p>
</div>



#### Graphically Demanding

Using the minimum and recommended requirements variables can help us determine whether a game is graphically demanding. However, the reason why we had to scrape GPU data month by month is because games have different release dates. We can't use GPU data from 2019 to evaluate a game from 2017 because GPUs from 2019 are higher performing than GPUs from 2017.

Replace company name from names:
Steam would sometimes include multiple GPUs in their minimum/recommended hardware section. When two or more GPUs from the same company are listed, Steam would only include the full title of one GPU and the partial title for the other GPU. For example, assume two Nvidia cards, Nvidia Geforce GTX 1070 and Nvidia Geforce GTX 1080 are listed. Steam would list this as Nvidia GeForce GTX 1070 / GTX 1080. So removing "Nvidia GeForce" would allow us to search for the GTX 1080 part as well. 


```python
files = './GPU_Data/' + dates_months + '_75_ptile.csv'

def replace_company_name(str_in):
    str_in = str_in.replace('NVIDIA GeForce', '').strip()
    str_in = str_in.replace('AMD Radeon', '').strip()
    str_in = str_in.replace('Intel', '').strip()
    return str_in
```


```python
df_game = df_game.sort_values(by='date')
df_game
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>name</th>
      <th>date</th>
      <th>num_reviews</th>
      <th>pct_like</th>
      <th>num_positive_reviews</th>
      <th>minimum</th>
      <th>recommended</th>
      <th>popular_revs</th>
      <th>popular_pos_revs</th>
      <th>popular_pct_likes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>98</th>
      <td>https://store.steampowered.com/app/418370/Resi...</td>
      <td>Resident Evil 7 Biohazard</td>
      <td>2017/01</td>
      <td>20879</td>
      <td>93</td>
      <td>19417</td>
      <td>NVIDIA® GeForce® GTX 760 or AMD Radeon™ R7 260...</td>
      <td>NVIDIA® GeForce® GTX 1060 with 3GB VRAM</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1435</th>
      <td>https://store.steampowered.com/app/556180/Myst...</td>
      <td>Mysterium: A Psychic Clue Game</td>
      <td>2017/01</td>
      <td>230</td>
      <td>78</td>
      <td>179</td>
      <td>Intel HD Graphics 3000</td>
      <td>Intel HD Graphics 4000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>955</th>
      <td>https://store.steampowered.com/app/571880/Ange...</td>
      <td>Angels with Scaly Wings</td>
      <td>2017/01</td>
      <td>1039</td>
      <td>98</td>
      <td>1018</td>
      <td>DirectX Compatible Card</td>
      <td>DirectX Compatible Card</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>807</th>
      <td>https://store.steampowered.com/app/559610/Love...</td>
      <td>Love Ribbon</td>
      <td>2017/01</td>
      <td>1396</td>
      <td>93</td>
      <td>1298</td>
      <td>DirectX or OpenGL compatible card</td>
      <td>DirectX or OpenGL compatible card</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1257</th>
      <td>https://store.steampowered.com/app/568320/Pict...</td>
      <td>Pictopix</td>
      <td>2017/01</td>
      <td>499</td>
      <td>96</td>
      <td>479</td>
      <td>Shader Model 2.0</td>
      <td>Shader Model 2.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1253</th>
      <td>https://store.steampowered.com/app/1201400/Not...</td>
      <td>Not For Broadcast: Prologue</td>
      <td>2019/12</td>
      <td>510</td>
      <td>92</td>
      <td>469</td>
      <td>Dedicated video card is required</td>
      <td>Dedicated video card is required</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>https://store.steampowered.com/app/393420/Hurt...</td>
      <td>Hurtworld</td>
      <td>2019/12</td>
      <td>21386</td>
      <td>75</td>
      <td>16039</td>
      <td>GeForce 460/Radeon HD 5850/Intel HD 4600 with ...</td>
      <td>GeForce 660/Radeon HD 6970</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>936</th>
      <td>https://store.steampowered.com/app/601220/Zup_...</td>
      <td>Zup! F</td>
      <td>2019/12</td>
      <td>1071</td>
      <td>97</td>
      <td>1038</td>
      <td>Intel HD Graphics</td>
      <td>Intel HD Graphics</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>236</th>
      <td>https://store.steampowered.com/app/1066780/Tra...</td>
      <td>Transport Fever 2</td>
      <td>2019/12</td>
      <td>7742</td>
      <td>84</td>
      <td>6503</td>
      <td>NVIDIA GeForce GTX 560 or AMD Radeon HD 7850, ...</td>
      <td>NVIDIA GeForce GTX 1060 or AMD Radeon RX 580, ...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>242</th>
      <td>https://store.steampowered.com/app/351290/SURV...</td>
      <td>SURVIVAL: Postapocalypse Now</td>
      <td>2019/12</td>
      <td>7614</td>
      <td>54</td>
      <td>4111</td>
      <td>NVIDIA GeForce 470 GTX or AMD Radeon 6870 HD s...</td>
      <td>NVIDIA GeForce 470 GTX or AMD Radeon 6870 HD s...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1493 rows × 11 columns</p>
</div>




```python
df_game.set_index(np.array(range(0, df_game.shape[0])), inplace=True)
df_GPU = pd.read_csv(files[0])
df_GPU['GPU Name'] = df_GPU['GPU Name'].apply(lambda x: replace_company_name(x))
df_GPU['Change (%)'] = df_GPU['Change (%)'].apply(lambda x: float(x.replace('%', '')))
change_pct = df_GPU['Change (%)'].sum()
previous_date = df_game.loc[0, 'date']

min_demanding = np.zeros(df_game.shape[0], dtype=int)
rec_demanding = np.zeros(df_game.shape[0], dtype=int)
gpu_change_pct = []

count = 0

for x in range(0, df_game.shape[0]):
    if (df_game.loc[x, 'date'] != previous_date):
        df_GPU = pd.read_csv(files[count+1])
        df_GPU['GPU Name'] = df_GPU['GPU Name'].apply(lambda x: replace_company_name(x))
        df_GPU['Change (%)'] = df_GPU['Change (%)'].apply(lambda x: float(x.replace('%', '')))
        count += 1
        previous_date = df_game.loc[x, 'date']
        change_pct = df_GPU['Change (%)'].sum()
        
    gpu_change_pct.append(change_pct)
    for name in df_GPU['GPU Name']:
        if name in df_game.loc[x, 'minimum']:
            min_demanding[x] = 1
            rec_demanding[x] = 1
            break
        elif name in df_game.loc[x, 'recommended']:
            rec_demanding[x] = 1
```

Now that we've determined which games are and aren't demanding, we can create columns for these values


```python
df_game = df_game.assign(min_demanding=min_demanding, rec_demanding=rec_demanding, change_pct=gpu_change_pct)
df_game.drop(columns=['url', 'num_reviews', 'pct_like', 'num_positive_reviews', 'minimum', 'recommended'], inplace=True)
df_game
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>date</th>
      <th>popular_revs</th>
      <th>popular_pos_revs</th>
      <th>popular_pct_likes</th>
      <th>min_demanding</th>
      <th>rec_demanding</th>
      <th>change_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resident Evil 7 Biohazard</td>
      <td>2017/01</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mysterium: A Psychic Clue Game</td>
      <td>2017/01</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angels with Scaly Wings</td>
      <td>2017/01</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Love Ribbon</td>
      <td>2017/01</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pictopix</td>
      <td>2017/01</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>Not For Broadcast: Prologue</td>
      <td>2019/12</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>1489</th>
      <td>Hurtworld</td>
      <td>2019/12</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>1490</th>
      <td>Zup! F</td>
      <td>2019/12</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>1491</th>
      <td>Transport Fever 2</td>
      <td>2019/12</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>1492</th>
      <td>SURVIVAL: Postapocalypse Now</td>
      <td>2019/12</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.22</td>
    </tr>
  </tbody>
</table>
<p>1493 rows × 8 columns</p>
</div>




```python
df_game.isnull().any()
```




    name                 False
    date                 False
    popular_revs         False
    popular_pos_revs     False
    popular_pct_likes    False
    min_demanding        False
    rec_demanding        False
    change_pct           False
    dtype: bool




```python
df_final = pd.DataFrame({'date':[],'popular_revs':[],'popular_pos_revs':[],'popular_pct_likes':[],'min_demanding':[],'rec_demanding':[],'change_pct':[]})
dates = df_game['date'].unique()

temp = df_game[df_game['date'] == '2017/01']
for x in dates:
    temp = df_game[df_game['date'] == x]
    try:
        pop_revs = (temp['popular_revs'].value_counts()[1])
    except:
        pop_revs = 0
    
    try:
        pop_pos_revs = (temp['popular_pos_revs'].value_counts()[1])
    except:
        pop_pos_revs = 0
        
    try:
        pop_pct_likes = (temp['popular_pct_likes'].value_counts()[1])
    except:
        pop_pct_likes = 0
        
    try:
        min_demand = (temp['min_demanding'].value_counts()[1])
    except:
        min_demand = 0

    try:
        rec_demand = (temp['rec_demanding'].value_counts()[1])
    except:
        rec_demand = 0
        
    change_pct = (temp['change_pct'].unique()[0])
    
    df2 = pd.DataFrame({'date': x,
    'popular_revs':[pop_revs],
    'popular_pos_revs':[pop_pos_revs],
    'popular_pct_likes':[pop_pct_likes],
    'min_demanding':[min_demand],
    'rec_demanding':[rec_demand],
    'change_pct':[change_pct]}, index=[0])
    
    df_final = df_final.append(df2, ignore_index=True)

df_final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>popular_revs</th>
      <th>popular_pos_revs</th>
      <th>popular_pct_likes</th>
      <th>min_demanding</th>
      <th>rec_demanding</th>
      <th>change_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017/01</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>5.800000e-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017/02</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>1.680000e+00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017/03</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>35.0</td>
      <td>2.0</td>
      <td>11.0</td>
      <td>2.250000e+00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017/04</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>29.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>1.140000e+00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017/05</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>37.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>-2.000000e-02</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017/06</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>3.260000e+00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2017/07</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>23.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>-3.469447e-18</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017/08</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>40.0</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>5.370000e+00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2017/09</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>41.0</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>2.380000e+00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2017/10</td>
      <td>12.0</td>
      <td>14.0</td>
      <td>27.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>8.360000e+00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2017/11</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>27.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>2.920000e+00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2017/12</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>26.0</td>
      <td>5.0</td>
      <td>12.0</td>
      <td>2.160000e+00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2018/01</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>24.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.665335e-16</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2018/02</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>-1.580000e+00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2018/03</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>43.0</td>
      <td>1.0</td>
      <td>13.0</td>
      <td>8.300000e-01</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2018/04</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>22.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>-6.150000e+00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2018/05</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>37.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>2.300000e-01</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2018/06</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>22.0</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>2.100000e-01</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2018/07</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>24.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>-2.300000e-01</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2018/08</td>
      <td>14.0</td>
      <td>15.0</td>
      <td>44.0</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>1.960000e+00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2018/09</td>
      <td>15.0</td>
      <td>16.0</td>
      <td>50.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>5.000000e-01</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2018/10</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>34.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>5.600000e-01</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2018/11</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>48.0</td>
      <td>3.0</td>
      <td>16.0</td>
      <td>-3.600000e-01</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2018/12</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>32.0</td>
      <td>6.0</td>
      <td>21.0</td>
      <td>3.900000e-01</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2019/01</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>35.0</td>
      <td>3.0</td>
      <td>15.0</td>
      <td>-2.800000e-01</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2019/02</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>32.0</td>
      <td>4.0</td>
      <td>17.0</td>
      <td>2.320000e+00</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2019/03</td>
      <td>11.0</td>
      <td>10.0</td>
      <td>36.0</td>
      <td>3.0</td>
      <td>15.0</td>
      <td>1.700000e-01</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2019/04</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>42.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>7.800000e-01</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2019/05</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>42.0</td>
      <td>3.0</td>
      <td>24.0</td>
      <td>5.800000e-01</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2019/06</td>
      <td>13.0</td>
      <td>12.0</td>
      <td>47.0</td>
      <td>4.0</td>
      <td>15.0</td>
      <td>6.900000e-01</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2019/07</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>35.0</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>-1.340000e+00</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2019/08</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>46.0</td>
      <td>2.0</td>
      <td>14.0</td>
      <td>6.300000e-01</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2019/09</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>41.0</td>
      <td>1.0</td>
      <td>15.0</td>
      <td>3.520000e+00</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2019/10</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>47.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>5.900000e-01</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2019/11</td>
      <td>9.0</td>
      <td>6.0</td>
      <td>54.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>1.610000e+00</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2019/12</td>
      <td>11.0</td>
      <td>9.0</td>
      <td>43.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.220000e+00</td>
    </tr>
  </tbody>
</table>
</div>



Now, we have our final dataset.

#### Summary:

The variables we will use for data analysis.

### Independent Variables
- popular_revs
- popular_pos_revs
- popular_pct_likes
- min_demanding
- rec_demanding

### Dependent Variable
- change_pct


The following information gives the definition for all of the variables contained in our dataframe.

#### Identification
date:
- Time frame we're looking at

#### Popularity 
popular_revs: 
- Number of games that are popular depending on the number of reviews

popular_pos_revs: 
- Number of games that are popular depending on the number of positive reviews

popular_pct_likes: 
- Number of games that are popular depending on the percetange of likes

#### Graphically Intensive
min_demanding: 
- Number of games that have demanding minimum requirements

rec_demanding: 
- Number of games that have demanding recommended requirements

#### Usage/sales
change_pct:
- The change in usage percentage for higher performing/latest GPUs for the month when the game was released.

# Data Analysis & Results

Our goal is to determine if there's a correlation between release of popular, graphically demanding games and GPU sales. We were unable to attain sales numbers of specific GPUs so as an alternative, we will be using the change in usage percentage. We know as sales for newer GPUs increase, it's likely that usage rate of newer GPUs also increase. There are some problems with this that we will address later in the issues section.

First, lets look at the distribution of usage percentage.


```python
plt.figure(figsize=(15,8))
sns.histplot(x=df_final['change_pct'], kde=True, stat="density", linewidth=0).set(xlabel='Change in Usage Percentage')
```




    [Text(0.5, 0, 'Change in Usage Percentage')]




    
![png](output_81_1.png)
    


Looking at the distribution for change in usage percentage of higher performing/newer GPUs, we can see that it is normal with a mean greater than 0 as well as some larger values in both the negative and positive directions. We can determine when these occur.


```python
plt.figure(figsize=(15,8))
ax = sns.lineplot(x='date', y='change_pct', data=df_final)
plt.xlabel('Date', fontsize = 13)
plt.ylabel('Change in usage percentage', fontsize = 13)
plt.title('Change in usage percentage plotted over time')
plt.xticks(fontsize=12, rotation=90)
plt.show()
```


    
![png](output_83_0.png)
    


We were able to identify where the large values from the distribution plot occurs, but more importantly, we can see that the change in usage percentage has periods of sudden increase and some periods show greater increase than others. Now that we know there exists periods where usage rate of higher performing/newer GPUs, we can begin to analyze if there's any correlation to our popular/demanding variables.

#### Looking for Patterns (Eyeball Test)

Lets see if any of the popularity variables show any similar patterns


```python
plt.figure(figsize=(15,8))
ax = sns.lineplot(x='date', y='popular_revs', data=df_final)

plt.xticks(fontsize=12, rotation=90)
plt.title('Popular games based on number of reviews', fontsize = 13)
plt.ylabel('Popular games', fontsize = 13)
plt.show()
```


    
![png](output_86_0.png)
    



```python
plt.figure(figsize=(15,8))
ax = sns.lineplot(x='date', y='popular_pos_revs', data=df_final)
plt.ylabel('Popular games', fontsize = 13)
plt.title('Popular games based on number of positive reviews')
plt.xticks(fontsize=12, rotation=90)
plt.show()
```


    
![png](output_87_0.png)
    



```python
plt.figure(figsize=(15,8))
ax = sns.lineplot(x='date', y='popular_pct_likes', data=df_final)
plt.title('Popular games based on percentage of likes')
plt.ylabel('Popular games', fontsize = 13)
plt.xticks(fontsize=12, rotation=90)
plt.show()
```


    
![png](output_88_0.png)
    


#### What do we find about the popularity graphs?

None of them show any significant patterns to the change in usage percentage graph.

There's a few small differences between the number of games with a popular number of reviews and the number of games with a popular number of positive reviews, but in general, they share a similar pattern. Their slope throughout the years looks to be horizontal, but it isn't entirely clear by simply looking at the graphs.

The graph for number of games with a popular percentage of likes has an upward sloping pattern.


```python
plt.figure(figsize=(15,8))
ax = sns.lineplot(x='date', y='min_demanding', data=df_final)
plt.title('Number of games that have demanding minimum requirements over time')
plt.ylabel('Number of games', fontsize = 13)

plt.xticks(fontsize=12, rotation=90)
plt.show()
```


    
![png](output_90_0.png)
    



```python
plt.figure(figsize=(15,8))
ax = sns.lineplot(x='date', y='rec_demanding', data=df_final)
plt.title('Number of games that have demanding recommended requirements over time')
plt.ylabel('Number of games', fontsize = 13)

plt.xticks(fontsize=12, rotation=90)
plt.show()
```


    
![png](output_91_0.png)
    


#### What do we find about the demanding graphs?

There doesn't appear to be any significant patterns relating to change in usage percentage.

The graph for the number of games with a demanding minimum requirement looks to have a horizontal slope.

The slope of the graph for the number of games with a demanding recommended requirement looks to be positive, but it's not entirely clear since it begins to drop significantly in the later months.

### Linear Regression

Instead of simply looking for patterns, linear regression models can be used to look for correlations. We can start by plotting the graph to get a visual, then take a look at the Ordinary Least Squares (OLS) regression results for more details.


```python
plt.figure(figsize=(15,8))
plt.title('Change in usage percentage plotted over popular reviews')

sns.regplot(x='popular_revs', y='change_pct', data=df_final)
plt.xlabel('Popular reviews', fontsize = 13)
plt.ylabel('Change in usage percentage', fontsize = 13)
```




    Text(0, 0.5, 'Change in usage percentage')




    
![png](output_94_1.png)
    



```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_revs', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.038</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.010</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.337</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.256</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:18</td>     <th>  Log-Likelihood:    </th> <td> -78.499</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   161.0</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    34</td>      <th>  BIC:               </th> <td>   164.2</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>    <td>   -0.4226</td> <td>    1.305</td> <td>   -0.324</td> <td> 0.748</td> <td>   -3.075</td> <td>    2.230</td>
</tr>
<tr>
  <th>popular_revs</th> <td>    0.1398</td> <td>    0.121</td> <td>    1.156</td> <td> 0.256</td> <td>   -0.106</td> <td>    0.385</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>10.732</td> <th>  Durbin-Watson:     </th> <td>   1.523</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.005</td> <th>  Jarque-Bera (JB):  </th> <td>  27.842</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.180</td> <th>  Prob(JB):          </th> <td>9.00e-07</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.293</td> <th>  Cond. No.          </th> <td>    38.7</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The p-value for popular_revs is 0.256 meaning that there is a %25.6 chance that popular_revs has no affect on change_pct (dependent variable). The relationship is not statistically significant (p > 0.05). R-squared is 0.038, which means that %3.8 of the variance in change_pct can be explained by our model


```python
plt.figure(figsize=(15,8))
plt.title('Change in usage percentage plotted over popular positive reviews')

sns.regplot(x='popular_pos_revs', y='change_pct', data=df_final)
plt.xlabel('Popular positive reviews', fontsize = 13)
plt.ylabel('Change in usage percentage', fontsize = 13)
```




    Text(0, 0.5, 'Change in usage percentage')




    
![png](output_97_1.png)
    



```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_pos_revs', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.091</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.064</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3.406</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td>0.0737</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:19</td>     <th>  Log-Likelihood:    </th> <td> -77.474</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   158.9</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    34</td>      <th>  BIC:               </th> <td>   162.1</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>        <td>   -1.1260</td> <td>    1.219</td> <td>   -0.923</td> <td> 0.362</td> <td>   -3.604</td> <td>    1.352</td>
</tr>
<tr>
  <th>popular_pos_revs</th> <td>    0.2077</td> <td>    0.113</td> <td>    1.846</td> <td> 0.074</td> <td>   -0.021</td> <td>    0.436</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 9.041</td> <th>  Durbin-Watson:     </th> <td>   1.474</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.011</td> <th>  Jarque-Bera (JB):  </th> <td>  19.551</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.018</td> <th>  Prob(JB):          </th> <td>5.68e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.610</td> <th>  Cond. No.          </th> <td>    37.3</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The p-value for popular_pos_revs is 0.074 meaning that there is a %7.4 chance that popular_pos_revs has no affect on change_pct (dependent variable). So, the relationship is not statistically significant (p > 0.05). R-squared is 0.091, which means that %9.1 of the variance in change_pct can be explained by our model.


```python
plt.figure(figsize=(15,8))
sns.regplot(x='popular_pct_likes', y='change_pct', data=df_final)
plt.xlabel('Percentage of likes', fontsize = 13)
plt.ylabel('Change in usage percentage', fontsize = 13)
plt.title('Change in usage percentage plotted over percentage of likes')
```




    Text(0.5, 1.0, 'Change in usage percentage plotted over percentage of likes')




    
![png](output_100_1.png)
    



```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_pct_likes', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.006</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.023</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.2195</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.642</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:19</td>     <th>  Log-Likelihood:    </th> <td> -79.077</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   162.2</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    34</td>      <th>  BIC:               </th> <td>   165.3</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>    0.3993</td> <td>    1.388</td> <td>    0.288</td> <td> 0.775</td> <td>   -2.422</td> <td>    3.221</td>
</tr>
<tr>
  <th>popular_pct_likes</th> <td>    0.0180</td> <td>    0.038</td> <td>    0.469</td> <td> 0.642</td> <td>   -0.060</td> <td>    0.096</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>12.082</td> <th>  Durbin-Watson:     </th> <td>   1.583</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.002</td> <th>  Jarque-Bera (JB):  </th> <td>  29.875</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.437</td> <th>  Prob(JB):          </th> <td>3.26e-07</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.376</td> <th>  Cond. No.          </th> <td>    135.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The p-value for popular_pct_likes is 0.642 meaning that there is a %64.2 chance that popular_pct_likes has no affect on change_pct (dependent variable). Therefore, the relationship is not statistically significant (p > 0.05). R-squared is 0.006, which means that %0.6 of the variance in change_pct can be explained by our model. As we can see, effects of popular_pct_likes on our dependent variable is minimal.


```python
plt.figure(figsize=(15,8))
sns.regplot(x='min_demanding', y='change_pct', data=df_final)
plt.xlabel('# of games that have demanding min requirements', fontsize = 13)
plt.ylabel('Change in usage percentage', fontsize = 13)
plt.title('Change in usage percentage plotted over # games that have demanding min requirements')
```




    Text(0.5, 1.0, 'Change in usage percentage plotted over # games that have demanding min requirements')




    
![png](output_103_1.png)
    



```python
outcome, predictors = patsy.dmatrices('change_pct ~ min_demanding', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.007</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.022</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.2428</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.625</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:19</td>     <th>  Log-Likelihood:    </th> <td> -79.065</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   162.1</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    34</td>      <th>  BIC:               </th> <td>   165.3</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>    1.3183</td> <td>    0.701</td> <td>    1.881</td> <td> 0.069</td> <td>   -0.106</td> <td>    2.743</td>
</tr>
<tr>
  <th>min_demanding</th> <td>   -0.1268</td> <td>    0.257</td> <td>   -0.493</td> <td> 0.625</td> <td>   -0.650</td> <td>    0.396</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>11.593</td> <th>  Durbin-Watson:     </th> <td>   1.623</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.003</td> <th>  Jarque-Bera (JB):  </th> <td>  27.933</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.402</td> <th>  Prob(JB):          </th> <td>8.60e-07</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.240</td> <th>  Cond. No.          </th> <td>    5.63</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The p-value for min_demanding is 0.625 meaning that there is a %62.5 chance that min_demanding has no affect on change_pct (dependent variable). The relationship is not statistically significant (p > 0.05). R-squared is 0.007, which means that %0.7 of the variance in change_pct can be explained by our model. Effect of min_demanding on our dependent variable is minimal


```python
plt.figure(figsize=(15,8))
sns.regplot(x='rec_demanding', y='change_pct', data=df_final)
plt.xlabel('# of games that have demanding recommended requirements', fontsize = 13)
plt.ylabel('Change in usage percentage', fontsize = 13)
plt.title('Change in usage percentage plotted over # games that have demanding recommended requirements')
```




    Text(0.5, 1.0, 'Change in usage percentage plotted over # games that have demanding recommended requirements')




    
![png](output_106_1.png)
    



```python
outcome, predictors = patsy.dmatrices('change_pct ~ rec_demanding', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.007</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.023</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.2256</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.638</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:20</td>     <th>  Log-Likelihood:    </th> <td> -79.074</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   162.1</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    34</td>      <th>  BIC:               </th> <td>   165.3</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>    1.4652</td> <td>    0.997</td> <td>    1.469</td> <td> 0.151</td> <td>   -0.562</td> <td>    3.492</td>
</tr>
<tr>
  <th>rec_demanding</th> <td>   -0.0366</td> <td>    0.077</td> <td>   -0.475</td> <td> 0.638</td> <td>   -0.193</td> <td>    0.120</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>11.014</td> <th>  Durbin-Watson:     </th> <td>   1.669</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.004</td> <th>  Jarque-Bera (JB):  </th> <td>  29.253</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.205</td> <th>  Prob(JB):          </th> <td>4.45e-07</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.397</td> <th>  Cond. No.          </th> <td>    34.8</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The p-value for rec_demanding is 0.638 meaning that there is a %63.8 chance that rec_demanding has no affect on change_pct (dependent variable). The relationship is not statistically significant (p > 0.05). R-squared is 0.007, which means that %0.7 of the variance in change_pct can be explained by our model

#### Summary from Simple Linear Regression

Individually, none of them show correlation to change in usage percentage. The 95% confidence interval for all of our popularity variables and demanding variables contained 0. This can be seen in the summaries and in the shaded blue regions of the graphs.

One thing of note is the R-squared value for all models is relatively small.

## Multivariate Linear Regression

Although none of our dependent variables show correlation to the change in number of usage percentage individually, We can try to show correlation through the combination of multiple independent variables.

#### Popular Variables

#### Independent variables
- popular_revs
- popular_pct_likes


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_revs + popular_pct_likes', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.038</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.020</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.6515</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.528</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:20</td>     <th>  Log-Likelihood:    </th> <td> -78.496</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   163.0</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    33</td>      <th>  BIC:               </th> <td>   167.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   -0.4899</td> <td>    1.629</td> <td>   -0.301</td> <td> 0.765</td> <td>   -3.804</td> <td>    2.824</td>
</tr>
<tr>
  <th>popular_revs</th>      <td>    0.1365</td> <td>    0.131</td> <td>    1.041</td> <td> 0.306</td> <td>   -0.130</td> <td>    0.403</td>
</tr>
<tr>
  <th>popular_pct_likes</th> <td>    0.0029</td> <td>    0.041</td> <td>    0.071</td> <td> 0.944</td> <td>   -0.081</td> <td>    0.086</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>10.796</td> <th>  Durbin-Watson:     </th> <td>   1.519</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.005</td> <th>  Jarque-Bera (JB):  </th> <td>  27.762</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.209</td> <th>  Prob(JB):          </th> <td>9.37e-07</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.282</td> <th>  Cond. No.          </th> <td>    164.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



P-values of both of our independent variables are greater than 0.05. So, there is no statistically significant relationship between our independent variables and dependent variable. R-squared is 0.038, which means that %3.8 of the variance in change_pct can be explained by our model.

#### Independent variables
- popular_pct_likes
- popular_pos_revs


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_pos_revs + popular_pct_likes', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.091</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.036</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.656</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.206</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:20</td>     <th>  Log-Likelihood:    </th> <td> -77.472</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   160.9</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    33</td>      <th>  BIC:               </th> <td>   165.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   -1.0580</td> <td>    1.583</td> <td>   -0.668</td> <td> 0.509</td> <td>   -4.279</td> <td>    2.163</td>
</tr>
<tr>
  <th>popular_pos_revs</th>  <td>    0.2102</td> <td>    0.120</td> <td>    1.754</td> <td> 0.089</td> <td>   -0.034</td> <td>    0.454</td>
</tr>
<tr>
  <th>popular_pct_likes</th> <td>   -0.0027</td> <td>    0.039</td> <td>   -0.069</td> <td> 0.946</td> <td>   -0.082</td> <td>    0.077</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 9.063</td> <th>  Durbin-Watson:     </th> <td>   1.477</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.011</td> <th>  Jarque-Bera (JB):  </th> <td>  19.670</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.007</td> <th>  Prob(JB):          </th> <td>5.35e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.621</td> <th>  Cond. No.          </th> <td>    164.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



P-values of both of our independent variables are greater than 0.05. So, there is no statistically significant relationship between our independent variables and dependent variable. R-squared is 0.091, which means that %9.1 of the variance in change_pct can be explained by our model

#### Independent variables
- popular_revs
- popular_pos_revs


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_revs + popular_pos_revs', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.148</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.096</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2.866</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td>0.0712</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:20</td>     <th>  Log-Likelihood:    </th> <td> -76.310</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   158.6</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    33</td>      <th>  BIC:               </th> <td>   163.4</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>        <td>   -0.5991</td> <td>    1.250</td> <td>   -0.479</td> <td> 0.635</td> <td>   -3.142</td> <td>    1.943</td>
</tr>
<tr>
  <th>popular_revs</th>     <td>   -0.4724</td> <td>    0.318</td> <td>   -1.485</td> <td> 0.147</td> <td>   -1.119</td> <td>    0.175</td>
</tr>
<tr>
  <th>popular_pos_revs</th> <td>    0.6292</td> <td>    0.305</td> <td>    2.066</td> <td> 0.047</td> <td>    0.009</td> <td>    1.249</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 6.712</td> <th>  Durbin-Watson:     </th> <td>   1.494</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.035</td> <th>  Jarque-Bera (JB):  </th> <td>  10.072</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.087</td> <th>  Prob(JB):          </th> <td> 0.00650</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.585</td> <th>  Cond. No.          </th> <td>    54.7</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Here we can see that the p-value of popular_pos_revs is slightly lower than 0.05, which was not the case when we used a single variable regression. This might indicate that the affect of popular_pos_revs might be significant when used together with popular_revs. However, we still need further analysis to confirm that

#### Independent variables
- All popular variables


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_revs + popular_pct_likes + popular_pos_revs', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.150</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.070</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.876</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.154</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:20</td>     <th>  Log-Likelihood:    </th> <td> -76.277</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   160.6</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    32</td>      <th>  BIC:               </th> <td>   166.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   -0.8203</td> <td>    1.563</td> <td>   -0.525</td> <td> 0.603</td> <td>   -4.005</td> <td>    2.364</td>
</tr>
<tr>
  <th>popular_revs</th>      <td>   -0.4891</td> <td>    0.330</td> <td>   -1.482</td> <td> 0.148</td> <td>   -1.161</td> <td>    0.183</td>
</tr>
<tr>
  <th>popular_pct_likes</th> <td>    0.0095</td> <td>    0.039</td> <td>    0.242</td> <td> 0.810</td> <td>   -0.071</td> <td>    0.090</td>
</tr>
<tr>
  <th>popular_pos_revs</th>  <td>    0.6353</td> <td>    0.310</td> <td>    2.049</td> <td> 0.049</td> <td>    0.004</td> <td>    1.267</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 6.397</td> <th>  Durbin-Watson:     </th> <td>   1.494</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.041</td> <th>  Jarque-Bera (JB):  </th> <td>   9.229</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.030</td> <th>  Prob(JB):          </th> <td> 0.00991</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.480</td> <th>  Cond. No.          </th> <td>    171.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



P-values of our independent variables are greater than 0.05 (with one being ~ 0.05). So, there is no statistically significant relationship between our independent variables and dependent variable. R-squared is 0.150, which means that %1.5 of the variance in change_pct can be explained by our model

#### Demanding Variables

We find little point in using min_demanding and rec_demanding together as independent variables because if a game has a demanding minimum requirement, then it also has a demanding recommended requirement. However, it is only one linear regression model and including it will not take long.

NOTE: In the multivariate regressions with popular and demanding variables combined, we will separate min_demanding and rec_demanding. In other words, none of the models will contain both min_demanding and rec_demanding.

#### Independent Variables
- min_demanding
- rec_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ min_demanding + rec_demanding', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.009</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.051</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.1576</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.855</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:20</td>     <th>  Log-Likelihood:    </th> <td> -79.022</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   164.0</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    33</td>      <th>  BIC:               </th> <td>   168.8</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>    1.5282</td> <td>    1.031</td> <td>    1.482</td> <td> 0.148</td> <td>   -0.570</td> <td>    3.626</td>
</tr>
<tr>
  <th>min_demanding</th> <td>   -0.0902</td> <td>    0.292</td> <td>   -0.309</td> <td> 0.759</td> <td>   -0.684</td> <td>    0.503</td>
</tr>
<tr>
  <th>rec_demanding</th> <td>   -0.0245</td> <td>    0.087</td> <td>   -0.281</td> <td> 0.781</td> <td>   -0.202</td> <td>    0.153</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>11.170</td> <th>  Durbin-Watson:     </th> <td>   1.653</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.004</td> <th>  Jarque-Bera (JB):  </th> <td>  27.781</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.315</td> <th>  Prob(JB):          </th> <td>9.28e-07</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.257</td> <th>  Cond. No.          </th> <td>    36.1</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



P-values of our independent variables are significantly greater than 0.05. So, there is no statistically significant relationship between our independent variables and dependent variable.

#### Summary of Multivariate Regression while Popular and Demanding variables are separated

No correlation could be found while keeping popular and demanding variables separate.

It doesn't seem that popularity of a video game nor demanding games correlates to change in usage percentage individually. However, popular and demanding video games may show a correlation to change in usage percentage.

#### min_demanding and Popular variables

##### Independent variables
- popular_revs
- min_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_revs + min_demanding', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.053</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.004</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.9242</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.407</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:21</td>     <th>  Log-Likelihood:    </th> <td> -78.212</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   162.4</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    33</td>      <th>  BIC:               </th> <td>   167.2</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>   -0.1636</td> <td>    1.362</td> <td>   -0.120</td> <td> 0.905</td> <td>   -2.934</td> <td>    2.607</td>
</tr>
<tr>
  <th>popular_revs</th>  <td>    0.1569</td> <td>    0.124</td> <td>    1.265</td> <td> 0.215</td> <td>   -0.095</td> <td>    0.409</td>
</tr>
<tr>
  <th>min_demanding</th> <td>   -0.1891</td> <td>    0.260</td> <td>   -0.728</td> <td> 0.472</td> <td>   -0.718</td> <td>    0.340</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>10.750</td> <th>  Durbin-Watson:     </th> <td>   1.508</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.005</td> <th>  Jarque-Bera (JB):  </th> <td>  24.195</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.358</td> <th>  Prob(JB):          </th> <td>5.57e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.952</td> <th>  Cond. No.          </th> <td>    41.0</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



P-values of our independent variables are greater than 0.05. So, there is no statistically significant relationship between our independent variables and dependent variable. R-squared is 0.053, which means that %5.3 of the variance in change_pct can be explained by our model

##### Independent variables
- popular_pct_likes
- min_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_pct_likes + min_demanding', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.012</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.048</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.2058</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.815</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:21</td>     <th>  Log-Likelihood:    </th> <td> -78.970</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   163.9</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    33</td>      <th>  BIC:               </th> <td>   168.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>    0.7256</td> <td>    1.585</td> <td>    0.458</td> <td> 0.650</td> <td>   -2.500</td> <td>    3.951</td>
</tr>
<tr>
  <th>popular_pct_likes</th> <td>    0.0163</td> <td>    0.039</td> <td>    0.418</td> <td> 0.679</td> <td>   -0.063</td> <td>    0.096</td>
</tr>
<tr>
  <th>min_demanding</th>     <td>   -0.1163</td> <td>    0.262</td> <td>   -0.444</td> <td> 0.660</td> <td>   -0.649</td> <td>    0.416</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>12.378</td> <th>  Durbin-Watson:     </th> <td>   1.585</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.002</td> <th>  Jarque-Bera (JB):  </th> <td>  28.321</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.530</td> <th>  Prob(JB):          </th> <td>7.08e-07</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.214</td> <th>  Cond. No.          </th> <td>    153.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



P-values of our independent variables are greater than 0.05. So, there is no statistically significant relationship between our independent variables and dependent variable. R-squared is 0.012, which means that %1.2 of the variance in change_pct can be explained by our model

##### Independent variables
- popular_pos_revs
- min_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_pos_revs + min_demanding', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.120</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.067</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2.254</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.121</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:21</td>     <th>  Log-Likelihood:    </th> <td> -76.889</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   159.8</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    33</td>      <th>  BIC:               </th> <td>   164.5</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>        <td>   -0.8470</td> <td>    1.247</td> <td>   -0.679</td> <td> 0.502</td> <td>   -3.383</td> <td>    1.689</td>
</tr>
<tr>
  <th>popular_pos_revs</th> <td>    0.2401</td> <td>    0.117</td> <td>    2.059</td> <td> 0.047</td> <td>    0.003</td> <td>    0.477</td>
</tr>
<tr>
  <th>min_demanding</th>    <td>   -0.2666</td> <td>    0.255</td> <td>   -1.045</td> <td> 0.304</td> <td>   -0.786</td> <td>    0.253</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 8.334</td> <th>  Durbin-Watson:     </th> <td>   1.466</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.016</td> <th>  Jarque-Bera (JB):  </th> <td>  14.541</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.258</td> <th>  Prob(JB):          </th> <td>0.000696</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.070</td> <th>  Cond. No.          </th> <td>    39.1</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



P-value for popular_pos_revs is barely lower than 0.05. So, we might say that there is probably no significant relationship between our variables. R-squared is 0.120, which means that %12 of the variance in change_pct can be explained by our model

##### Independent variables
- popular_revs
- popular_pct_likes
- min_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_revs + min_demanding + popular_pct_likes', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.053</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.036</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.5986</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.621</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:21</td>     <th>  Log-Likelihood:    </th> <td> -78.210</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   164.4</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    32</td>      <th>  BIC:               </th> <td>   170.8</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   -0.1037</td> <td>    1.727</td> <td>   -0.060</td> <td> 0.952</td> <td>   -3.622</td> <td>    3.415</td>
</tr>
<tr>
  <th>popular_revs</th>      <td>    0.1599</td> <td>    0.136</td> <td>    1.175</td> <td> 0.249</td> <td>   -0.117</td> <td>    0.437</td>
</tr>
<tr>
  <th>min_demanding</th>     <td>   -0.1919</td> <td>    0.268</td> <td>   -0.716</td> <td> 0.479</td> <td>   -0.738</td> <td>    0.354</td>
</tr>
<tr>
  <th>popular_pct_likes</th> <td>   -0.0024</td> <td>    0.042</td> <td>   -0.058</td> <td> 0.954</td> <td>   -0.088</td> <td>    0.083</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>10.646</td> <th>  Durbin-Watson:     </th> <td>   1.512</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.005</td> <th>  Jarque-Bera (JB):  </th> <td>  24.128</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.337</td> <th>  Prob(JB):          </th> <td>5.76e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.954</td> <th>  Cond. No.          </th> <td>    173.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



P-values of our independent variables are greater than 0.05. So, there is no statistically significant relationship between our independent variables and dependent variable. R-squared is 0.053, which means that %5.3 of the variance in change_pct can be explained by our model

##### Independent variables
- popular_pos_revs
- popular_pct_likes
- min_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_pos_revs + min_demanding + popular_pct_likes', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.122</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.040</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.485</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.237</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:21</td>     <th>  Log-Likelihood:    </th> <td> -76.847</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   161.7</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    32</td>      <th>  BIC:               </th> <td>   168.0</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   -0.5596</td> <td>    1.648</td> <td>   -0.340</td> <td> 0.736</td> <td>   -3.917</td> <td>    2.798</td>
</tr>
<tr>
  <th>popular_pos_revs</th>  <td>    0.2517</td> <td>    0.126</td> <td>    2.001</td> <td> 0.054</td> <td>   -0.004</td> <td>    0.508</td>
</tr>
<tr>
  <th>min_demanding</th>     <td>   -0.2804</td> <td>    0.264</td> <td>   -1.063</td> <td> 0.296</td> <td>   -0.818</td> <td>    0.257</td>
</tr>
<tr>
  <th>popular_pct_likes</th> <td>   -0.0108</td> <td>    0.040</td> <td>   -0.272</td> <td> 0.787</td> <td>   -0.092</td> <td>    0.070</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 8.054</td> <th>  Durbin-Watson:     </th> <td>   1.480</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.018</td> <th>  Jarque-Bera (JB):  </th> <td>  14.346</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.171</td> <th>  Prob(JB):          </th> <td>0.000767</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.074</td> <th>  Cond. No.          </th> <td>    172.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



P-values of our independent variables are greater than 0.05. So, there is no statistically significant relationship between our independent variables and dependent variable. R-squared is 0.122, which means that %12.2 of the variance in change_pct can be explained by our model

##### Independent variables
- popular_revs
- popular_pos_revs
- min_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_revs + min_demanding + popular_pos_revs', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.194</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.118</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2.560</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td>0.0723</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:21</td>     <th>  Log-Likelihood:    </th> <td> -75.321</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   158.6</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    32</td>      <th>  BIC:               </th> <td>   165.0</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>        <td>   -0.1654</td> <td>    1.276</td> <td>   -0.130</td> <td> 0.898</td> <td>   -2.765</td> <td>    2.434</td>
</tr>
<tr>
  <th>popular_revs</th>     <td>   -0.5439</td> <td>    0.319</td> <td>   -1.707</td> <td> 0.098</td> <td>   -1.193</td> <td>    0.105</td>
</tr>
<tr>
  <th>min_demanding</th>    <td>   -0.3382</td> <td>    0.252</td> <td>   -1.344</td> <td> 0.188</td> <td>   -0.851</td> <td>    0.174</td>
</tr>
<tr>
  <th>popular_pos_revs</th> <td>    0.7341</td> <td>    0.311</td> <td>    2.361</td> <td> 0.024</td> <td>    0.101</td> <td>    1.367</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.270</td> <th>  Durbin-Watson:     </th> <td>   1.543</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.072</td> <th>  Jarque-Bera (JB):  </th> <td>   5.837</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.184</td> <th>  Prob(JB):          </th> <td>  0.0540</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.938</td> <th>  Cond. No.          </th> <td>    57.3</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The p-value for popular_pos_revs is below 0.05, which was not the case when we used single variable linear regression. This could mean that popular_pos_revs might have a significant affect on our dependent variable in this given context.

##### Independent variables
- All popular variables
- min_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_revs + min_demanding + popular_pos_revs + popular_pct_likes', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.194</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.090</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.860</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.142</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:21</td>     <th>  Log-Likelihood:    </th> <td> -75.320</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   160.6</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    31</td>      <th>  BIC:               </th> <td>   168.6</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   -0.1937</td> <td>    1.620</td> <td>   -0.120</td> <td> 0.906</td> <td>   -3.498</td> <td>    3.111</td>
</tr>
<tr>
  <th>popular_revs</th>      <td>   -0.5456</td> <td>    0.329</td> <td>   -1.657</td> <td> 0.108</td> <td>   -1.217</td> <td>    0.126</td>
</tr>
<tr>
  <th>min_demanding</th>     <td>   -0.3370</td> <td>    0.259</td> <td>   -1.301</td> <td> 0.203</td> <td>   -0.865</td> <td>    0.191</td>
</tr>
<tr>
  <th>popular_pos_revs</th>  <td>    0.7345</td> <td>    0.316</td> <td>    2.324</td> <td> 0.027</td> <td>    0.090</td> <td>    1.379</td>
</tr>
<tr>
  <th>popular_pct_likes</th> <td>    0.0011</td> <td>    0.039</td> <td>    0.029</td> <td> 0.977</td> <td>   -0.079</td> <td>    0.082</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.264</td> <th>  Durbin-Watson:     </th> <td>   1.543</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.072</td> <th>  Jarque-Bera (JB):  </th> <td>   5.803</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.189</td> <th>  Prob(JB):          </th> <td>  0.0549</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.930</td> <th>  Cond. No.          </th> <td>    180.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Again, the p-value of popular_pos_revs is below 0.05. We might infer that popular_pos_revs have a significant impact when used in this given context. Also, the value R-squared went up compared to other results, which can mean that we can explain more of the variance by this set of independent variables

#### rec_demanding and Popular variables

##### Independent variables
- popular_revs
- rec_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_revs + rec_demanding', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.063</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.006</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.113</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.341</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:22</td>     <th>  Log-Likelihood:    </th> <td> -78.018</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   162.0</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    33</td>      <th>  BIC:               </th> <td>   166.8</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>    0.0548</td> <td>    1.402</td> <td>    0.039</td> <td> 0.969</td> <td>   -2.797</td> <td>    2.906</td>
</tr>
<tr>
  <th>popular_revs</th>  <td>    0.1824</td> <td>    0.129</td> <td>    1.412</td> <td> 0.167</td> <td>   -0.080</td> <td>    0.445</td>
</tr>
<tr>
  <th>rec_demanding</th> <td>   -0.0766</td> <td>    0.081</td> <td>   -0.945</td> <td> 0.352</td> <td>   -0.242</td> <td>    0.088</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>10.382</td> <th>  Durbin-Watson:     </th> <td>   1.601</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.006</td> <th>  Jarque-Bera (JB):  </th> <td>  27.021</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.023</td> <th>  Prob(JB):          </th> <td>1.36e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.244</td> <th>  Cond. No.          </th> <td>    63.3</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



P-values of our independent variables are greater than 0.05. So, there is no statistically significant relationship between our independent variables and dependent variable. R-squared is 0.063, which means that %6.3 of the variance in change_pct can be explained by our model

##### Independent variables
- popular_pct_likes
- rec_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_pct_likes + rec_demanding', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.030</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.029</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.5059</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.608</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:22</td>     <th>  Log-Likelihood:    </th> <td> -78.649</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   163.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    33</td>      <th>  BIC:               </th> <td>   168.0</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>    0.5857</td> <td>    1.408</td> <td>    0.416</td> <td> 0.680</td> <td>   -2.279</td> <td>    3.451</td>
</tr>
<tr>
  <th>popular_pct_likes</th> <td>    0.0414</td> <td>    0.047</td> <td>    0.887</td> <td> 0.381</td> <td>   -0.053</td> <td>    0.136</td>
</tr>
<tr>
  <th>rec_demanding</th>     <td>   -0.0833</td> <td>    0.094</td> <td>   -0.891</td> <td> 0.379</td> <td>   -0.274</td> <td>    0.107</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>11.804</td> <th>  Durbin-Watson:     </th> <td>   1.650</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.003</td> <th>  Jarque-Bera (JB):  </th> <td>  27.263</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.462</td> <th>  Prob(JB):          </th> <td>1.20e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.162</td> <th>  Cond. No.          </th> <td>    144.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



P-values of our independent variables are greater than 0.05. So, there is no statistically significant relationship between our independent variables and dependent variable. R-squared is 0.030, which means that %3.0 of the variance in change_pct can be explained by our model

##### Independent variables
- popular_pos_revs
- rec_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_pos_revs + rec_demanding', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.135</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.083</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2.580</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td>0.0910</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:22</td>     <th>  Log-Likelihood:    </th> <td> -76.578</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   159.2</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    33</td>      <th>  BIC:               </th> <td>   163.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>        <td>   -0.5071</td> <td>    1.298</td> <td>   -0.391</td> <td> 0.699</td> <td>   -3.148</td> <td>    2.134</td>
</tr>
<tr>
  <th>popular_pos_revs</th> <td>    0.2664</td> <td>    0.120</td> <td>    2.215</td> <td> 0.034</td> <td>    0.022</td> <td>    0.511</td>
</tr>
<tr>
  <th>rec_demanding</th>    <td>   -0.1023</td> <td>    0.079</td> <td>   -1.298</td> <td> 0.203</td> <td>   -0.263</td> <td>    0.058</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 9.387</td> <th>  Durbin-Watson:     </th> <td>   1.599</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.009</td> <th>  Jarque-Bera (JB):  </th> <td>  18.970</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.268</td> <th>  Prob(JB):          </th> <td>7.60e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.516</td> <th>  Cond. No.          </th> <td>    61.1</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The p-value of popular_pos_revs is below 0.05. So, the effect of this independent variable could be significant in this given context. We can also see that the value of R-squared went up compared to other results.

##### Independent variables
- popular_revs
- popular_pct_likes
- rec_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_revs + rec_demanding + popular_pct_likes', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.074</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>  -0.012</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>  0.8581</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.473</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:22</td>     <th>  Log-Likelihood:    </th> <td> -77.800</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   163.6</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    32</td>      <th>  BIC:               </th> <td>   169.9</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   -0.4418</td> <td>    1.623</td> <td>   -0.272</td> <td> 0.787</td> <td>   -3.748</td> <td>    2.864</td>
</tr>
<tr>
  <th>popular_revs</th>      <td>    0.1656</td> <td>    0.133</td> <td>    1.243</td> <td> 0.223</td> <td>   -0.106</td> <td>    0.437</td>
</tr>
<tr>
  <th>rec_demanding</th>     <td>   -0.1062</td> <td>    0.095</td> <td>   -1.123</td> <td> 0.270</td> <td>   -0.299</td> <td>    0.086</td>
</tr>
<tr>
  <th>popular_pct_likes</th> <td>    0.0295</td> <td>    0.047</td> <td>    0.625</td> <td> 0.537</td> <td>   -0.067</td> <td>    0.126</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>10.242</td> <th>  Durbin-Watson:     </th> <td>   1.600</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.006</td> <th>  Jarque-Bera (JB):  </th> <td>  24.665</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.193</td> <th>  Prob(JB):          </th> <td>4.41e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.037</td> <th>  Cond. No.          </th> <td>    173.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



P-values of our independent variables are greater than 0.05. So, there is no statistically significant relationship between our independent variables and dependent variable. R-squared is 0.074, which means that %7.4 of the variance in change_pct can be explained by our model

##### Independent variables:
- popular_pos_revs
- popular_pct_likes
- rec_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_pos_revs + rec_demanding + popular_pct_likes', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.148</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.068</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.846</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td> 0.159</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:22</td>     <th>  Log-Likelihood:    </th> <td> -76.320</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   160.6</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    32</td>      <th>  BIC:               </th> <td>   167.0</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   -1.0803</td> <td>    1.557</td> <td>   -0.694</td> <td> 0.493</td> <td>   -4.252</td> <td>    2.092</td>
</tr>
<tr>
  <th>popular_pos_revs</th>  <td>    0.2567</td> <td>    0.122</td> <td>    2.102</td> <td> 0.043</td> <td>    0.008</td> <td>    0.505</td>
</tr>
<tr>
  <th>rec_demanding</th>     <td>   -0.1341</td> <td>    0.092</td> <td>   -1.454</td> <td> 0.156</td> <td>   -0.322</td> <td>    0.054</td>
</tr>
<tr>
  <th>popular_pct_likes</th> <td>    0.0304</td> <td>    0.045</td> <td>    0.679</td> <td> 0.502</td> <td>   -0.061</td> <td>    0.121</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 8.261</td> <th>  Durbin-Watson:     </th> <td>   1.625</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.016</td> <th>  Jarque-Bera (JB):  </th> <td>  15.840</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.069</td> <th>  Prob(JB):          </th> <td>0.000363</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.247</td> <th>  Cond. No.          </th> <td>    173.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The p-value of popular_pos_revs is slighyly under 0.05, which is a recurring pattern so far in our analysis.

##### Independent variables
- popular_revs
- popular_pos_revs
- rec_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_revs + rec_demanding + popular_pos_revs', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.193</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.117</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2.543</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td>0.0736</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:23</td>     <th>  Log-Likelihood:    </th> <td> -75.344</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   158.7</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    32</td>      <th>  BIC:               </th> <td>   165.0</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>        <td>    0.0237</td> <td>    1.322</td> <td>    0.018</td> <td> 0.986</td> <td>   -2.668</td> <td>    2.716</td>
</tr>
<tr>
  <th>popular_revs</th>     <td>   -0.4738</td> <td>    0.314</td> <td>   -1.507</td> <td> 0.142</td> <td>   -1.114</td> <td>    0.167</td>
</tr>
<tr>
  <th>rec_demanding</th>    <td>   -0.1027</td> <td>    0.077</td> <td>   -1.328</td> <td> 0.194</td> <td>   -0.260</td> <td>    0.055</td>
</tr>
<tr>
  <th>popular_pos_revs</th> <td>    0.6894</td> <td>    0.305</td> <td>    2.264</td> <td> 0.030</td> <td>    0.069</td> <td>    1.310</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 7.670</td> <th>  Durbin-Watson:     </th> <td>   1.634</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.022</td> <th>  Jarque-Bera (JB):  </th> <td>  11.342</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.323</td> <th>  Prob(JB):          </th> <td> 0.00344</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.673</td> <th>  Cond. No.          </th> <td>    75.2</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The p-value of popular_pos_revs is lower than 0.05, which can mean that it's effects might be significant. We need further analysis.

##### Independent variables
- All popular variables
- rec_demanding


```python
outcome, predictors = patsy.dmatrices('change_pct ~ popular_revs + rec_demanding + popular_pos_revs + popular_pct_likes', df_final)
mod = sm.OLS(outcome, predictors)
res = mod.fit()
res.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>change_pct</td>    <th>  R-squared:         </th> <td>   0.223</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.123</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2.226</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 10 Dec 2021</td> <th>  Prob (F-statistic):</th>  <td>0.0891</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>23:55:23</td>     <th>  Log-Likelihood:    </th> <td> -74.649</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   159.3</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    31</td>      <th>  BIC:               </th> <td>   167.2</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   -0.8109</td> <td>    1.518</td> <td>   -0.534</td> <td> 0.597</td> <td>   -3.907</td> <td>    2.286</td>
</tr>
<tr>
  <th>popular_revs</th>      <td>   -0.5614</td> <td>    0.323</td> <td>   -1.737</td> <td> 0.092</td> <td>   -1.221</td> <td>    0.098</td>
</tr>
<tr>
  <th>rec_demanding</th>     <td>   -0.1546</td> <td>    0.090</td> <td>   -1.713</td> <td> 0.097</td> <td>   -0.339</td> <td>    0.029</td>
</tr>
<tr>
  <th>popular_pos_revs</th>  <td>    0.7518</td> <td>    0.309</td> <td>    2.435</td> <td> 0.021</td> <td>    0.122</td> <td>    1.381</td>
</tr>
<tr>
  <th>popular_pct_likes</th> <td>    0.0494</td> <td>    0.045</td> <td>    1.105</td> <td> 0.278</td> <td>   -0.042</td> <td>    0.141</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 5.217</td> <th>  Durbin-Watson:     </th> <td>   1.773</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.074</td> <th>  Jarque-Bera (JB):  </th> <td>   5.936</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.123</td> <th>  Prob(JB):          </th> <td>  0.0514</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.974</td> <th>  Cond. No.          </th> <td>    180.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



The p-value of popular_pos_revs is lower than 0.05, and our R-squared is 0.223, which might mean that we are onto something. We are going to further analyze these variables using variance inflation factor

#### Summary of Multivariate Regressions while Popular and Demanding variables were combined

Above we have 14 different combinations of popular and demanding variables. These are the correlations we were able to find:

#### min_demanding and Popular variables Summary

##### Independent variables
- popular_revs
- popular_pos_revs
- min_demanding

When using popular_revs and popular_pos_revs only, we get a correlation for popular_pos_revs with a slope of 0.7341 and an R-square value of 0.194.

##### Independent variables
- All popular variables
- min_demanding

When using all popular variables, we get a correlation for popular_pos_revs with a slope of 0.7345 and an R-square value of 0.194.

While using these two combinations of variables, we were able to find a correlation between change in usage percentage and number of games with popular number of positive reviews.

#### rec_demanding and Popular variables Summary

##### Independent variables
- popular_pos_revs
- rec_demanding

When using popular_pos_revs only, we get a correlation for popular_pos_revs with a slope of 0.2664, R-squared value of 0.135

##### Independent variables
- popular_revs
- popular_pos_revs
- rec_demanding

When using popular_revs and popular_pos_revs only, we get a correlation for popular_pos_revs with a slope of 0.6894, R-squared value of 0.193

##### Independent variables
- All popular variables
- rec_demanding

When using all popular variables, we get a correlation for popular pos_revs with a slope of 0.7518, R-squared value of 0.223

While using these three combinations of variables, we were able to find a correlation between change in usage percentage and number of games with popular number of positive reviews.

#### Summary of Multivariate Regression

We weren't able to find any correlations when keeping popular and demanding variables separate, however, when combining these two, we were able to find a correlation. Specifically with the number of games with popular number of positive reviews. 

In multiple regression models with combined popular and demanding variables, we were able to find a slight positive correlation between change in usage percentage and number of games with popular number of positive reviews. 

## Is the correlation significant?

In the multivariate regression, we found correlation for four models. One concern is that our independent variables are collinear and to check for collinearity, we looked at variance inflation factor (VIF). The higher the VIF of an independent variable, the more correlated it is with the other independent variables. It's common that a VIF > 10 is an indicator of multicollinearity and so we will use that as our threshold.

We can check the VIF of the four models that gave us a correlation.

Independent variables:
- popular_revs
- popular_pos_revs
- min_demanding


```python
vif = pd.DataFrame()
var = df_final[['popular_revs', 'popular_pos_revs', 'min_demanding']]
vif['variables'] = var.columns
vif['VIF'] = [variance_inflation_factor(var.values, i) for i in range(len(var.columns))]
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variables</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>popular_revs</td>
      <td>88.909308</td>
    </tr>
    <tr>
      <th>1</th>
      <td>popular_pos_revs</td>
      <td>94.463866</td>
    </tr>
    <tr>
      <th>2</th>
      <td>min_demanding</td>
      <td>3.657769</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,8))
graph = sns.barplot(x='VIF', y='variables', data=vif, palette= ['blue', 'orange', 'green'])
graph.axvline(10, c = 'k', linestyle = '--')

plt.xlabel('Variance Inflation Factor (VIF)', fontsize = 13)
plt.ylabel('Variables', fontsize = 13)
plt.title('First Set of Independent Variables Against VIF', fontsize = 13)
plt.show()
```


    
![png](output_179_0.png)
    


VIF for popular variables are too high! These values are way beyond the acceptable VIF values. 

Independent variables:
- All popular variables
- min_demanding


```python
vif = pd.DataFrame()
var = df_final[['popular_revs', 'popular_pos_revs', 'min_demanding', 'popular_pct_likes']]
vif['variables'] = var.columns
vif['VIF'] = [variance_inflation_factor(var.values, i) for i in range(len(var.columns))]
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variables</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>popular_revs</td>
      <td>100.091775</td>
    </tr>
    <tr>
      <th>1</th>
      <td>popular_pos_revs</td>
      <td>94.554075</td>
    </tr>
    <tr>
      <th>2</th>
      <td>min_demanding</td>
      <td>3.659232</td>
    </tr>
    <tr>
      <th>3</th>
      <td>popular_pct_likes</td>
      <td>10.472384</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,8))

graph = sns.barplot(x='VIF', y='variables', data=vif, palette = ['blue', 'orange', 'green', 'red'])
graph.axvline(10, c = 'k', linestyle = '--')

plt.xlabel('Variance Inflation Factor (VIF)', fontsize = 13)
plt.ylabel('Variables', fontsize = 13)
plt.title('Second Set of Independent Variables Against VIF', fontsize = 13)
plt.show()
```


    
![png](output_183_0.png)
    


Popular variables too high!

##### Independent variables:
- popular_revs
- popular_pos_revs
- rec_demanding


```python
vif = pd.DataFrame()
var = df_final[['popular_revs', 'popular_pos_revs', 'rec_demanding']]
vif['variables'] = var.columns
vif['VIF'] = [variance_inflation_factor(var.values, i) for i in range(len(var.columns))]
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variables</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>popular_revs</td>
      <td>89.009600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>popular_pos_revs</td>
      <td>90.509127</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rec_demanding</td>
      <td>7.271300</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,8))

graph = sns.barplot(x='VIF', y='variables', data=vif, palette = ['blue', 'orange', 'pink'])
graph.axvline(10, c = 'k', linestyle = '--')

plt.xlabel('Variance Inflation Factor (VIF)', fontsize = 13)
plt.ylabel('Variables', fontsize = 13)
plt.title('Third Set of Independent Variables Against VIF', fontsize = 13)
plt.show()
```


    
![png](output_187_0.png)
    


Popular variables too high!

#### Independent variables:
- popular_pos_revs
- rec_demanding


```python
vif = pd.DataFrame()
var = df_final[['popular_pos_revs', 'rec_demanding']]
vif['variables'] = var.columns
vif['VIF'] = [variance_inflation_factor(var.values, i) for i in range(len(var.columns))]
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variables</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>popular_pos_revs</td>
      <td>7.196824</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rec_demanding</td>
      <td>7.196824</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,8))

graph = sns.barplot(x='VIF', y='variables', data=vif, palette = ['orange', 'pink'])
graph.axvline(10, c= 'k', linestyle = '--')

plt.xlabel('Variance Inflation Factor (VIF)', fontsize = 13)
plt.ylabel('Variables', fontsize = 13)
plt.title('Fourth Set of Independent Variables Against VIF', fontsize = 13)
plt.show()
```


    
![png](output_191_0.png)
    


Passed the VIF test! VIF values are still relatively high.

#### Independent variables:
- All popular variables
- rec_demanding


```python
vif = pd.DataFrame()
var = df_final[['popular_revs', 'popular_pos_revs', 'popular_pct_likes', 'rec_demanding']]
vif['variables'] = var.columns
vif['VIF'] = [variance_inflation_factor(var.values, i) for i in range(len(var.columns))]
vif
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variables</th>
      <th>VIF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>popular_revs</td>
      <td>100.883778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>popular_pos_revs</td>
      <td>92.716103</td>
    </tr>
    <tr>
      <th>2</th>
      <td>popular_pct_likes</td>
      <td>16.437015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rec_demanding</td>
      <td>11.417294</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,8))

graph = sns.barplot(x='VIF', y='variables', data=vif, palette = ['blue', 'orange', 'red', 'pink'])
graph.axvline(10, c= 'k', linestyle = '--')

plt.xlabel('Variance Inflation Factor (VIF)', fontsize = 13)
plt.ylabel('Variables', fontsize = 13)
plt.title('Fifth Set of Independent Variables Against VIF', fontsize = 13)
plt.show()
```


    
![png](output_195_0.png)
    


All variables too high!

#### Variance Inflation Factor Results

Checking VIF values for the four models that showed correlation indicates collinearity issue for all four multivariate regression models. 

##### Popular Variables

The VIF for the popular variables in all four models are too high! 

##### Demanding Variables

The VIF for min_demanding is within acceptable range for both models.
The VIF for rec_demanding is relatively high in both models. It exceeds the threshold in the model with all popular variables.

#### Variance Inflation Factor Summary

From these results, it's highly probable that the correlations seen in the multivarite regression models are a result of collinearity and are not significant.

# Ethics & Privacy

The data sets we used are publicly available through the steam hardware and software website. Althought steam only give us the latest data, we were able to capture previous survey data by accessing wayback machine. Wayback machine is an initiative of the Internet Archive under 501(c)(3). Wayback machine and the no person can alter it once it have been captured. Steam is the largest digital distribution platfrom for PC gaming. Steam collect its user hardware and calculate the change compare to previous month.  

Regarding privacy, the datasets we used do not contain sensitive personal information and therefore do not infringe on any person’s privacy. User are chosen randomly and require the user consent before practicipe. All of the data are keep anonymous and will not be associate to the user steam account.  

Some biases that could be present in our analysis would come from the usere who are practiciping. Since most user who are on steam are aroung 20's, they are more likely to buy a new hardware once a while.

Another bias is using the popular variables we decided to go with. Popular has many definitions such as anticipation/pre-order buys, largest player base, and more. We beleived anticipation/pre-order could have been a better measurement of popularity in this case. However, when deciding for popularity variable measures, we chose variables that are easier to attain. We decided to do this because of two reasons. Firstly, we believe our variables can still give us a good measurement on GPU sales, just not as good as anticipation/pre-order buys and secondly, we are unsure if enough information on anticipation/pre-order buys was attainable since games on "Steam" don't have pre-order. 

# Conclusion & Discussion

#### Conclusion:
GPUs can be used for many reasons, such as machine learning or video editing to name a couple, but GPUs are also utilized to play video games. Many video games contain graphics that require the use of GPUs to render. Since GPUs and video games have this relation to them, we hypothesized that the release of a popular, demanding video game is positively correlated to an increase in GPU sales. 

In an attempt to find the effect of popular and demanding games on GPU sales, we took a look at "Steam", the largest online market for PC video games. Web scraping "Steam" yielded several measures on popularity of a game and how demanding a game is, as well as the change in usage percentages of GPUs per month, which became our measure for GPU sales.

We began by performing an eyeball test to look for any noticeable patterns between our popular or demanding measures with the change in usage percentages. Each variable was plotted on separate line graphs to show patterns over time. Results showed that two of the popular variables, games with a popular number of reviews and games with a popular number of positive reviews, had almost identical patterns, but no other noticeable patterns.

Unable to find anything relating our popular and demanding variables to the variable of interest from the eyeball test, we turned to simple linear regression. Results from all simple linear models contained 0 in their 95% confidence interval which means no correlation could be found.

Still with no correlation found, we tried using multivariate regression. Our multivariate regression models found correlation for five models. To test whether the correlations were actually significant, VIF tests were performed on each model and four models that showed correlation were determined to be insignificant. The VIF tests for each of these four OLS regression models showed severe collinearity in popular variables for all four models. In addition to this, the VIF values for games with demanding recommended requirements were high in both models it was tested in, which could indicate that there's correlation between games with demanding recommended requirements and popularity. The one model that didn't fail the VIF test still had high VIF values.

From the result of the analysis, one correlation could be found between popular and demanding video games on GPU sales. The release of a video game that achieves a popular positive review correlates with GPU sales.

#### Limitations:
We were able to find a correlation in our analysis, but this may have been due to some data limitations. One such limitation comes from our games dataset. Originally, the games dataset had over 80000 entries, but only 1493 of them went into the constructing the final dataset. In comparison, 24132 games were released on "Steam" over the 3 year interval this analysis was restricted to[7]. Since the values for the independent variables are counts of video games, the amount of games that go into the analysis can have substantial impact on the independent variables. In addition to this, the games dataset only considers "Steam" games which means console video games are not included. Consoles also make up a considerable portion of video games and the exclusion of console video games also directly impact the independent variables. 

Another limitation comes from our measure of variables. We were unable to attain GPU sales numbers and as a substitute, used the change in usage percentage of GPUs. A problem with this is we're only able to make correlation claims about popular, demanding games and change in usage percentage and from this, and therefore, the results of this analysis is only one possible indication to GPU sales. For popularity, a measurement that we considered, but also could not find the data for is the anticipation of a video game.

#### Discussion:
Our analysis found one correlation between popular and demanding games with GPU sales which could indicate that the demand for accelerated 3D rendering has not diminished over time. However, from the VIF tests, there is potentially a correlation between popularity and the demand of a video game. If a correlation can be found with GPU sales and popularity, then it's possible that popularity of video games could be a mediator variable between the graphical demand of video games and GPU sales. More analysis is needed in order to say there's a correlation for certain.

# Team Contributions

Each team member was assigned a specific portion of the report to write:


- Cameron VanderTuig: Background & Prior Work, Hypothesis, Dataset(s), Setup, Double Checking, Video
- Erdogan Ergit: Background & Prior Work, Documentation, Data Visualization
- Henry Chan: Background & Prior Work, Dataset(s), Setup, Ethics & Privacy, Code Optimization, Double Checking
- Wilson Tan: Overview, Background & Prior Work, Dataset(s), Setup, Data Cleaning, Data Analysis & Results, Conclusion & Discussion, Double Checking, Documentation


```python

```
