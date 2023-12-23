import requests
from bs4 import BeautifulSoup
import csv

def scrape_website(url):
    # Send an HTTP request and get the HTML content
    response = requests.get(url)
    html_content = response.content

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract data using BeautifulSoup methods
    # Find tables on the Wikipedia page
    tables = soup.find_all('table', class_='wikitable')

    # Save extracted data to a CSV file
    with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Title', 'Link'])

        # Iterate through each table
        for table in tables:
            rows = table.find_all('tr')
            
            # Iterate through each row in the table
            for row in rows:
                columns = row.find_all(['th', 'td'])
                if columns:
                    # Assuming the first column is the title and the second column is the link
                    title = columns[0].text.strip()
                    link = columns[1].find('a')['href'] if columns[1].find('a') else ''
                    
                    # Convert relative link to absolute link
                    full_link = f'https://en.wikipedia.org{link}' if link else ''
                    
                    csv_writer.writerow([title, full_link])

    print("Scraping completed. Data saved to 'output.csv'.")

if __name__ == "__main__":
    # Replace the URL with the actual Wikipedia page URL you want to scrape
    website_url = 'https://en.wikipedia.org/wiki/Matrix_(mathematics)'
    
    # Call the function to scrape the website
    scrape_website(website_url)
