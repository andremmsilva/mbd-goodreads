## RQ3: How has “Romantasy” evolved?

### Intuition

“Romantasy” is the overlap between **Fantasy** and **Romance**. We want to see if this hybrid genre:

* grows in the **catalog** (what gets published), and/or
* grows in **reader activity** (what people add/interact with).

So we measure **supply** and **demand** over time.


## Data used

* **Books metadata** (`goodreads_books.parquet`): gives `publication_year` per `book_id`. 
* **Genres** (`goodreads_book_genres_initial.parquet`): for each book, a nested `genres` structure that stores **vote counts per genre label** (e.g., `romance`, `fantasy, paranormal`, etc.). 
* **Interactions** (`goodreads_interactions_dedup.parquet`): for each `(user_id, book_id)` we have `date_added`, which we use to extract a year and build a “reader demand” time series. 


## Operational definition of Romantasy (what counts as romantasy?)

We create a synthetic “genre” called **romantasy**.

A book is labeled **romantasy** if:

1. It has meaningful support for both genres:

* votes(`fantasy, paranormal`) ≥ **THRESH**
* votes(`romance`) ≥ **THRESH**

2. And the two genres are **reasonably balanced** (to avoid “mostly fantasy with a tiny romance tag”):

* `min(fantasy_votes, romance_votes) / max(fantasy_votes, romance_votes) ≥ 0.2`

Interpretation:

> romance must be at least ~20% as strong as fantasy (and vice versa), among users’ genre votes for that book.

This reduces false positives where a book is overwhelmingly one genre.


## How we detect genres and romantasy in Spark (presentation-friendly)

#### Step 1 — Build per-book genre membership

From the `genres` field (a map/struct of `{genre_name → vote_count}`), we create a long table:

* **(book_id, genre)** exists if vote_count ≥ THRESH

This gives us “book belongs to genre G” using a consistent threshold.

#### Step 2 — Add romantasy as just another “genre”

We compute per-book:

* `fantasy_votes = genres["fantasy, paranormal"]`
* `romance_votes = genres["romance"]`

If the book passes the threshold + balance rule, we add:

* **(book_id, "romantasy")**

Then we union this with the normal genres table.
From this point on, romantasy is treated exactly like any other genre.


## Supply trend (published books)

Goal: “How much is each genre's share in the catalog each year?”

1. Take books with valid `publication_year`
2. Join with `(book_id, genre)` membership
3. For each year:

* `n_books_total(year)` = number of books published that year
* `n_books_genre(year, genre)` = number of published books that belong to genre

We compute a share:

* `share_books_genre = n_books_genre / n_books_total`

Result: a time series of **genre share among published books**.


## Demand trend (user interactions)

Goal: “What genres do users engage with each year?”

1. From interactions, extract year from `date_added` (e.g., last 4 digits)
2. Join interactions with `(book_id, genre)` membership
3. For each year:

* `n_interactions_total(year)` = total interactions that year
* `n_interactions_genre(year, genre)` = interactions involving books in that genre

Compute:

* `share_interactions_genre = n_interactions_genre / n_interactions_total`

Result: a time series of **genre share among user interactions**.


## Why this is “big-data friendly”

* We only select the needed columns (`book_id`, `publication_year`, `date_added`, and `genres`)
* We convert the nested genres into a simple `(book_id, genre)` membership table once, then reuse it.
* We aggregate early to year-level shares, so the final plotting data is tiny.
* We write small CSV outputs (year × genre) that fit easily on the driver / local machine for plotting.


## How to read the plots

We plot two panels:

### 1) Supply (published books)

Shows whether romantasy is becoming a larger fraction of what gets published.

### 2) Demand (user interactions)

Shows whether readers increasingly add/interact with romantasy books over time.

To contextualize romantasy, we also plot the **top K genres** (by average share) and include romantasy even if it’s not top-K. Romantasy is visually highlighted (thicker line).


## Limitations to mention (one sentence each)

* **Interactions end around 2017**, so we do not capture the very recent BookTok-driven romantasy boom.
* **Genre votes are cumulative**, so a book’s genre classification reflects overall consensus, not necessarily what users thought in the first year of publication.


