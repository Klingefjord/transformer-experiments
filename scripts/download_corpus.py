import argparse
import os
import re
from bs4 import BeautifulSoup
import requests


def download_corpus(author_name: str, author_id: int) -> None:
    """Download the works of the given author from gutenberg.org."""
    page = requests.get(f"https://www.gutenberg.org/ebooks/author/{author_id}")
    soup = BeautifulSoup(page.content, "html.parser")

    # get the list of books
    books = []
    for link in soup.find_all("a", href=True):
        if re.search(r"/ebooks/\d+", link["href"]):
            books.append(link["href"])

    print(f"Found {len(books)} books for {author_name}")

    # download the books
    for i, book in enumerate(books):
        print(f"Downloading book {i + 1}/{len(books)}...")

        page = requests.get(f"https://www.gutenberg.org{book}")
        soup = BeautifulSoup(page.content, "html.parser")

        # make sure the language is english
        if soup.find("tr", attrs={"itemprop": "inLanguage"})["content"] != "en":
            print("Skipping non-english book")
            continue

        # get the book text
        link = soup.find("a", href=True, text="Plain Text UTF-8")
        text = requests.get(f"https://www.gutenberg.org{link['href']}").content.decode(
            "utf-8"
        )

        # remove the header and footer
        text = text.split("***\r\n\r\n\r\n\r\n\r\n")
        text = text[1] if len(text) > 1 else text[0]
        text = text.split("\r\n\r\n\r\n\r\n\r\n***")[0]

        # create the directory if needed
        if not os.path.exists(f"../data/{author_name}"):
            os.makedirs(f"../data/{author_name}")

        # save the book
        path = f"../data/{author_name}/{link['href'].split('/')[-1]}"
        with open(path, "w") as f:
            f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--author_name", default="dostoyevsky")
    parser.add_argument("--author_id", default=314)
    args = parser.parse_args()

    download_corpus(args.author_name, args.author_id)
