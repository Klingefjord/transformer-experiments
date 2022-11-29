import argparse
import os
import re
from bs4 import BeautifulSoup
import requests


def download_books(books: list, author_name: str) -> None:
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

        if link is None:
            print("Skipping book with no text...")
            continue

        resp = requests.get(f"https://www.gutenberg.org{link['href']}")
        text = requests.get(f"https://www.gutenberg.org{link['href']}").content.decode(
            "utf-8"
        )

        # remove the header and footer
        text = text.split("***\r\n\r\n\r\n\r\n\r\n")
        text = text[1] if len(text) > 1 else text[0]
        text = text.split("\r\n\r\n\r\n\r\n\r\n***")[0]

        # append the text to the file
        with open(f"./data/{author_name}.txt", "a") as f:
            f.write(text)


def get_books(author_name: str, route: str, books: list = []) -> list[str]:
    """Recursively get a list of links to the books of the given author"""

    # navigate to the right page
    page = requests.get(f"https://www.gutenberg.org{route}")
    soup = BeautifulSoup(page.content, "html.parser")

    # get the list of books
    for link in soup.find_all("a", href=True):
        if re.search(r"/ebooks/\d+", link["href"]):
            books.append(link["href"])

    # handle pagination by recursion
    next_page = soup.find("a", href=True, text="Next")
    if next_page:
        return get_books(author_name, next_page["href"], books)
    else:
        return books


def download_corpus(author_name: str, author_id: int) -> None:
    """Download the works of the given author from gutenberg.org and save it to ./data/{author_name}.txt"""

    # get the list of books
    books = get_books(author_name, f"/ebooks/author/{author_id}")
    print(f"Found {len(books)} books for {author_name}")

    # download the books
    download_books(books, author_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--author_name", default="dostoyevsky")
    parser.add_argument("--author_id", default=314)
    args = parser.parse_args()

    download_corpus(args.author_name, args.author_id)
