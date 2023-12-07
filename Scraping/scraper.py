import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime


# or use the path to your WebDriver: webdriver.Chrome('/path/to/chromedriver')
driver = webdriver.Chrome()

# Function to read the starting index from startAt.txt
# Purpose: recover progress if program ends abruptly
def read_start_index():
    with open('startAt.txt', 'r') as file:
        return int(file.read().strip())

# Function to save the DataFrame and update the start index
def save_dataframe(df, index, start_index):
    df.to_csv(f'data_{start_index}.csv', index=False)
    with open('startAt.txt', 'w') as file:
        file.write(str(index))

# Function to scrape data from InterPro
def scrape_interpro(accession):
    description, sequence = None, None

    # URLs for scraping
    desc_url = f"https://www.ebi.ac.uk/interpro/protein/reviewed/{accession}/#table"
    seq_url = f"https://www.ebi.ac.uk/interpro/protein/reviewed/{accession}/sequence/"

    # Scrape Description
    try:
        driver.get(desc_url)
        # Wait for the button to be present, and then click it if it exists
        try:
            load_more_button = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located(
                    (By.XPATH, "/html/body/div[1]/div/div[2]/div[3]/div/section/section/div/section[1]/div[1]/table/tbody/tr[5]/td[2]/button"))
            )
            load_more_button.click()
            # Wait a moment for the content to load after clicking
        except Exception as e:
            pass

        # Now scrape the description
        desc_elem = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located(
                (By.XPATH, "/html/body/div[1]/div/div[2]/div[3]/div/section/section/div/section[1]/div[1]/table/tbody/tr[5]/td[2]/div/div/div"))
        )
        description = desc_elem.text
    except Exception as e:
        pass

    # Scrape Sequence
    try:
        driver.get(seq_url)
        seq_div = WebDriverWait(driver, 8).until(
            EC.presence_of_element_located(
                (By.XPATH, "/html/body/div[1]/div/div[2]/div[3]/div/section/section/section/div[2]/div[1]/div[2]"))
        )
        span_elements = seq_div.find_elements(By.TAG_NAME, 'span')
        sequence = ''.join([span.text for span in span_elements])
    except Exception as e:
        print(f"Error scraping sequence for {accession}")

    return description, sequence


# Read the starting index
start_index = read_start_index()

# Load the CSV file
df = pd.read_csv('mice output.csv')

# Add new columns if they don't exist
for col in ['Description', 'Sequence']:
    if col not in df.columns:
        df[col] = None

# Iterate over the DataFrame starting from the start_index
for i in range(start_index, len(df)):
    accession = df.at[i, 'Accession']
    description, sequence = scrape_interpro(accession)
    df.at[i, 'Description'] = description
    df.at[i, 'Sequence'] = sequence

    if description == None or sequence == None:
        save_dataframe(df, i, start_index)
        print("------- MISSING DATA ", i, accession,
              df.at[i, 'Name'], len(sequence))
        continue

    # Save every 50 rows
    if (i - start_index) % 100 == 0 and i != start_index:
        save_dataframe(df, i, start_index)
        # Get the current time
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        print("   =======================>>>>>>>>> ", time_str, " || ",
              i, df.iloc[i]['Name'], len(sequence))


print("Process completed.")
