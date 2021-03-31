import requests
from bs4 import BeautifulSoup

from selenium import webdriver
import time

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument("--test-type")

driver = webdriver.Chrome(chrome_options=options)
driver.get('https://bbc.com/news/world')
time.sleep(15)

button = driver.find_element_by_xpath('/html/body/)
#button = driver.find_element_by_xpath('///div[@id="orb-banner"]/header[@class="o-touch"]/div/div/div[@class="gs-u-display-none gs-u-display-block@m nw-o-news-wide-navigation"]/nav[@class="nw-c-nav__wide"]/ul[@class="gs-o-list-ui--top-no-border nw-c-nav__wide-sections"][4]')
#button = button[0]
button.click()