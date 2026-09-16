"""Offline, exclusive-create final45 evidence preparation. Never calls a model."""
QUESTIONS = [{'question_id': 'F11',
  'question': 'What is the total runtime of all movies released before 2000?',
  'expected_answer': '674 minutes',
  'batch': 'metadata',
  'category': 'arithmetic_aggregation'},
 {'question_id': 'F12',
  'question': 'What is the average runtime of movies released in or after 2010? Round the answer '
              'to two decimal places.',
  'expected_answer': '131.50 minutes',
  'batch': 'metadata',
  'category': 'arithmetic_aggregation'},
 {'question_id': 'F13',
  'question': 'What is the average IMDb rating of all movies directed by Christopher Nolan? Round '
              'the answer to two decimal places.',
  'expected_answer': '8.73',
  'batch': 'metadata',
  'category': 'arithmetic_aggregation'},
 {'question_id': 'F14',
  'question': 'Among movies with runtime of at least 150 minutes, what is the average IMDb rating? '
              'Round the answer to two decimal places.',
  'expected_answer': '8.67',
  'batch': 'metadata',
  'category': 'arithmetic_aggregation'},
 {'question_id': 'F15',
  'question': 'Among movies with an IMDb rating of at least 8.5, what is the average runtime? '
              'Round the answer to two decimal places.',
  'expected_answer': '157.43 minutes',
  'batch': 'metadata',
  'category': 'arithmetic_aggregation'},
 {'question_id': 'F16',
  'question': 'Among genres appearing in at least 4 movies, which genre has the highest average '
              'IMDb rating? Round the average to two decimal places.',
  'expected_answer': 'Drama - 8.49',
  'batch': 'metadata',
  'category': 'genre_level_aggregation'},
 {'question_id': 'F17',
  'question': 'Among genres appearing in at least 4 movies, which genre has the lowest total '
              'runtime across all its movies? Report the genre and its total runtime in minutes.',
  'expected_answer': 'Animation - 426 minutes',
  'batch': 'metadata',
  'category': 'genre_level_aggregation'},
 {'question_id': 'F18',
  'question': 'Which genre appears in the greatest number of movies in the dataset? Report the '
              'genre and the number of movies.',
  'expected_answer': 'Drama - 11 movies',
  'batch': 'metadata',
  'category': 'genre_level_aggregation'},
 {'question_id': 'F19',
  'question': 'Among genres appearing in at least 4 movies, which genre has the highest average '
              'runtime? Round the answer to two decimal places.',
  'expected_answer': 'Fantasy - 154.25 minutes',
  'batch': 'metadata',
  'category': 'genre_level_aggregation'},
 {'question_id': 'F20',
  'question': 'Among genres appearing in at least 4 movies, which genre has the lowest average '
              'runtime? Round the answer to two decimal places.',
  'expected_answer': 'Animation - 106.50 minutes',
  'batch': 'metadata',
  'category': 'genre_level_aggregation'},
 {'question_id': 'F21',
  'question': 'Among genres appearing in at least 4 movies, which genre has the highest proportion '
              'of its movies released in or after 2010? Report the genre and the percentage, '
              'rounded to two decimal places.',
  'expected_answer': 'Sci-Fi - 75.00%',
  'batch': 'metadata',
  'category': 'genre_level_aggregation'},
 {'question_id': 'F22',
  'question': 'Among genres appearing in at least 4 movies, which genre contains the greatest '
              'number of movies with an IMDb rating of at least 8.5? Report the genre and the '
              'number of qualifying movies.',
  'expected_answer': 'Drama - 6 movies',
  'batch': 'metadata',
  'category': 'genre_level_aggregation'},
 {'question_id': 'F23',
  'question': 'Using all 20 movies, identify the movies whose runtime is greater than the mean '
              'runtime plus 1 population standard deviation. Calculate the mean and population '
              'standard deviation as intermediate steps. Return only the movie titles.',
  'expected_answer': 'Saving Private Ryan; The Lord of the Rings: The Fellowship of the Ring; The '
                     'Lord of the Rings: The Two Towers; The Wolf of Wall Street; Interstellar',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'},
 {'question_id': 'F24',
  'question': 'Using all 20 movies, identify the movies whose runtime is less than the mean '
              'runtime minus 1 population standard deviation. Calculate the mean and population '
              'standard deviation as intermediate steps. Return only the movie titles.',
  'expected_answer': 'Monsters Inc.; Toy Story 3; How to Train Your Dragon; Black Swan',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'},
 {'question_id': 'F25',
  'question': 'Using all 20 movies, identify the movies whose IMDb rating is greater than the mean '
              'IMDb rating plus 1 population standard deviation. Calculate the mean and population '
              'standard deviation as intermediate steps. Return only the movie titles.',
  'expected_answer': 'The Lord of the Rings: The Fellowship of the Ring; The Lord of the Rings: '
                     'The Two Towers; The Dark Knight',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'},
 {'question_id': 'F26',
  'question': 'Among Adventure movies only, which movies have runtime greater than the '
              'Adventure-movie mean runtime + 1 population standard deviation?',
  'expected_answer': 'The Lord of the Rings: The Fellowship of the Ring; The Lord of the Rings: '
                     'The Two Towers; Interstellar',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'},
 {'question_id': 'F27',
  'question': 'Among Adventure movies only, which movies have runtime less than the '
              'Adventure-movie mean runtime - 1 population standard deviation?',
  'expected_answer': 'Monsters Inc.; Toy Story 3; How to Train Your Dragon',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'},
 {'question_id': 'F28',
  'question': 'Among Drama movies only, which movies have runtime greater than the Drama-movie '
              'mean runtime + 1 population standard deviation?',
  'expected_answer': 'The Lord of the Rings: The Fellowship of the Ring; The Lord of the Rings: '
                     'The Two Towers',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'},
 {'question_id': 'F29',
  'question': 'Among Action movies only, which movies have runtime greater than the Action-movie '
              'mean runtime + 1 population standard deviation?',
  'expected_answer': 'The Dark Knight',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'},
 {'question_id': 'F30',
  'question': 'Among Action movies only, which movies have runtime less than the Action-movie mean '
              'runtime - 1 population standard deviation?',
  'expected_answer': 'How to Train Your Dragon',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'},
 {'question_id': 'F31',
  'question': 'What is the total runtime of all movies with an IMDb rating of at least 8.5?',
  'expected_answer': '1102 minutes',
  'batch': 'metadata',
  'category': 'arithmetic_aggregation'},
 {'question_id': 'F32',
  'question': 'What is the difference between the average runtime of movies released before 2010 '
              'and movies released in or after 2010? Report both averages and the difference, '
              'rounded to two decimal places.',
  'expected_answer': 'Before 2010: 140.17 minutes; 2010 or after: 131.50 minutes; Difference: 8.67 '
                     'minutes',
  'batch': 'metadata',
  'category': 'arithmetic_aggregation'},
 {'question_id': 'F33',
  'question': 'Among movies with runtime less than 120 minutes, what is the average IMDb rating? '
              'Round the answer to two decimal places.',
  'expected_answer': '8.21',
  'batch': 'metadata',
  'category': 'arithmetic_aggregation'},
 {'question_id': 'F34',
  'question': 'Among movies with IMDb ratings from 8.0 through 8.4 inclusive, what is the average '
              'runtime? Round the answer to two decimal places.',
  'expected_answer': '125.55 minutes',
  'batch': 'metadata',
  'category': 'arithmetic_aggregation'},
 {'question_id': 'F35',
  'question': 'What is the total runtime of all movies directed by Christopher Nolan?',
  'expected_answer': '434 minutes',
  'batch': 'metadata',
  'category': 'arithmetic_aggregation'},
 {'question_id': 'F36',
  'question': 'Among genres appearing in at least 4 movies, which genre has the largest runtime '
              'range, calculated as maximum runtime minus minimum runtime? Report the genre and '
              'its runtime range in minutes.',
  'expected_answer': 'Comedy - 88 minutes',
  'batch': 'metadata',
  'category': 'genre_level_aggregation'},
 {'question_id': 'F37',
  'question': 'Among genres appearing in at least 4 movies, which genre has the highest average '
              'release year? Round the average year to two decimal places.',
  'expected_answer': 'Sci-Fi - 2010.00',
  'batch': 'metadata',
  'category': 'genre_level_aggregation'},
 {'question_id': 'F38',
  'question': 'Among genres appearing in at least 4 movies, which genre contains the greatest '
              'number of movies released in or after 2010? Report the genre and the number of '
              'qualifying movies.',
  'expected_answer': 'Adventure - 5 movies',
  'batch': 'metadata',
  'category': 'genre_level_aggregation'},
 {'question_id': 'F39',
  'question': 'Among genres appearing in exactly 4 movies, which genre has the highest total '
              'runtime? Report the genre and its total runtime in minutes.',
  'expected_answer': 'Fantasy - 617 minutes',
  'batch': 'metadata',
  'category': 'genre_level_aggregation'},
 {'question_id': 'F40',
  'question': 'Compare Drama and Adventure movies. Which genre has the higher average IMDb rating, '
              'and what is the difference between their averages? Round averages and the '
              'difference to two decimal places.',
  'expected_answer': 'Drama - 8.49; Adventure - 8.24; Difference: 0.25',
  'batch': 'metadata',
  'category': 'genre_level_aggregation'},
 {'question_id': 'F41',
  'question': 'Among movies released before 2010, use the rule runtime > subset mean runtime + 1 '
              'population standard deviation. Which movies are runtime outliers?',
  'expected_answer': 'Saving Private Ryan; The Lord of the Rings: The Fellowship of the Ring; The '
                     'Lord of the Rings: The Two Towers',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'},
 {'question_id': 'F42',
  'question': 'Among movies released before 2010, use the rule runtime < subset mean runtime - 1 '
              'population standard deviation. Which movies are runtime outliers?',
  'expected_answer': 'Memento; Monsters Inc.',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'},
 {'question_id': 'F43',
  'question': 'Among movies released in or after 2010, use the rule runtime > subset mean runtime '
              '+ 1 population standard deviation. Which movies are runtime outliers?',
  'expected_answer': 'The Wolf of Wall Street; Interstellar',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'},
 {'question_id': 'F44',
  'question': 'Among movies released in or after 2010, use the rule runtime < subset mean runtime '
              '- 1 population standard deviation. Which movies are runtime outliers?',
  'expected_answer': 'Toy Story 3; How to Train Your Dragon',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'},
 {'question_id': 'F45',
  'question': 'Among Action movies only, use the rule IMDb rating > Action-movie mean rating + 1 '
              'population standard deviation. Which movies are rating outliers?',
  'expected_answer': 'The Dark Knight',
  'batch': 'metadata',
  'category': 'statistical_outlier_detection'}]

import argparse
import csv
import hashlib
import io
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

NOTE = 'Frozen benchmark metadata. Use the listed benchmark director and normalized genres for all calculations.'
ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')

def wrap(draw, text, font, width):
    lines = []
    for word in text.split():
        candidate = (lines[-1] + ' ' + word) if lines else word
        if lines and draw.textlength(candidate, font=font) <= width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-csv', type=Path, required=True)
    parser.add_argument('--source-db', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    parser.add_argument('--bold-font', type=Path, required=True)
    args = parser.parse_args()
    out = ROOT / 'evidence/imdb_20_final45'
    gt = ROOT / 'ground_truth/imdb_20_final45'
    pdf_path = ROOT / 'evidence/imdb_20/imdb_20_final_v2_metadata_cards.pdf'
    config_path = ROOT / 'datasets/imdb_20_final45.yaml'
    prompt_path = ROOT / 'prompts/imdb_20_final45_llm_only_zero_shot.txt'
    for p in [out, gt, pdf_path, config_path, prompt_path]:
        if p.exists():
            raise FileExistsError(f'Refusing to overwrite {p}')
    rows = list(csv.DictReader(io.StringIO(args.source_csv.read_text())))
    db = sqlite3.connect(args.source_db.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    assert len(rows) == db.execute('select count(*) from movies').fetchone()[0] == 20
    assert len({r['movie_id'] for r in rows}) == len({r['title'] for r in rows}) == 20
    for r in rows:
        original = dict(db.execute('select * from movies where movie_id=?', (r['movie_id'],)).fetchone())
        assert all(str(v) == r[k] for k, v in original.items())
        genres = r['genres'].split('; ')
        db_genres = [g[0] for g in db.execute('select genre from movie_genres where movie_id=?', (r['movie_id'],))]
        assert set(genres) == set(db_genres) and len(genres) == len(db_genres) == len(set(genres))
    db.close()
    original_csv = ROOT / 'evidence/imdb_20/focused_10_question_pdf_questions.csv'
    frozen = list(csv.DictReader(io.StringIO(original_csv.read_text())))
    original_q = json.loads((ROOT / 'evidence/imdb_20/questions.json').read_text())['questions']
    assert len(frozen) == len(original_q) == 10
    questions = frozen + QUESTIONS
    assert Counter(r['category'] for r in questions) == {'arithmetic_aggregation':15,'genre_level_aggregation':15,'statistical_outlier_detection':15}
    out.mkdir(); gt.mkdir(); (out / 'cards').mkdir()
    with (out / 'frozen_movies.csv').open('xb') as f:
        f.write(args.source_csv.read_bytes())
    with (out / 'focused_45_question_pdf_questions.csv').open('x', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['question_id','batch','category','question','expected_answer']);writer.writeheader();writer.writerows(questions)
    expanded = list(original_q)
    for r in frozen:
        old = ROOT / 'ground_truth/imdb_20' / (r['question_id']+'.csv')
        with (gt / old.name).open('xb') as f: f.write(old.read_bytes())
    for r in QUESTIONS:
        filename = r['question_id']+'.csv'
        with (gt / filename).open('x', newline='') as f:
            writer=csv.writer(f);writer.writerow(['answer_text']);writer.writerow([r['expected_answer']])
        expanded.append({'id':r['question_id'],'text':r['question'],'category':r['category'],'source_pdf':'metadata','answer_format':'Return only the final answer.','ground_truth_file':filename})
    write_json(out / 'questions.json', {'questions':expanded})
    font = lambda size: ImageFont.truetype(str(args.font),size)
    bold = lambda size: ImageFont.truetype(str(args.bold_font),size)
    document = pymupdf.open()
    records = []
    for r in rows:
        image=Image.new('RGB',(1200,900),'#FFFFFF');d=ImageDraw.Draw(image)
        ink='#142333';muted='#405363';accent='#23666B'
        d.rectangle((0,0,1200,12), fill=accent)
        title_lines=wrap(d,r['title'],bold(48),1040)
        assert len(title_lines)<=3
        for i,line in enumerate(title_lines):d.text((80,58+i*57),line,font=bold(48),fill=ink)
        d.line((80,248,1120,248),fill='#CDD8DE',width=2)
        for x,label,value in [(80,'Release Year',r['year']),(425,'Runtime',r['runtime_minutes']+' minutes'),(820,'IMDb Rating',r['imdb_rating'])]:
            d.text((x,278),label,font=font(27),fill=muted)
            assert d.textlength(value,font=bold(40)) < (320 if x!=425 else 365)
            d.text((x,322),value,font=bold(40),fill=ink)
        d.text((80,425),'Benchmark Director:',font=font(27),fill=muted)
        assert d.textlength(r['director'],font=bold(40))<1040
        d.text((80,470),r['director'],font=bold(40),fill=ink)
        d.text((80,565),'Normalized Genres:',font=font(27),fill=muted)
        genre_lines=wrap(d,r['genres'],bold(40),1040)
        assert len(genre_lines)<=2
        for i,line in enumerate(genre_lines):d.text((80,610+i*50),line,font=bold(40),fill=ink)
        d.line((80,756,1120,756),fill='#CDD8DE',width=2)
        note_lines=wrap(d,NOTE,font(25),1040)
        assert len(note_lines)<=3
        for i,line in enumerate(note_lines):d.text((80,784+i*32),line,font=font(25),fill=muted)
        card=out/'cards'/f"movie_{int(r['movie_id']):02d}.png"
        with card.open('xb') as f:image.save(f,format='PNG')
        page=document.new_page(width=1200,height=900)
        page.insert_image(page.rect,filename=str(card))
        records.append({'movie_id':r['movie_id'],'page_index':len(records),'fields':{k:r[k] for k in ['title','year','runtime_minutes','imdb_rating','director','genres']},'card':str(card.relative_to(ROOT)),'card_sha256':sha(card),'source_csv_database_agree':True,'layout_bounds_pass':True})
    document.set_metadata({'title':'IMDb-20 final v2: image-only normalized metadata','author':'Researcher-generated benchmark evidence','subject':'Frozen per-movie normalized metadata'})
    with pdf_path.open('xb') as f:f.write(document.tobytes(garbage=4,deflate=True))
    document.close()
    with config_path.open('x') as f:f.write('''name: imdb_20_final45
description: "45-question IMDb benchmark: image-only normalized metadata, final v2"
evidence_type: pdf_visual
eval_strategy: partial_credit
evidence_path: evidence/imdb_20/imdb_20_final_v2_metadata_cards.pdf
questions_file: evidence/imdb_20_final45/questions.json
ground_truth_dir: ground_truth/imdb_20_final45/
pdf_page_count: 20
page_layout: metadata_only
pages_per_image: 4
render_scale: 1.0
jpeg_quality: 75
question_count: 45
condition: image-only normalized metadata
''')
    with prompt_path.open('x') as f:f.write('''You are given image-only normalized metadata for a frozen IMDb movie benchmark.
The cards are researcher-generated evidence, not unaltered IMDb screenshots.
Use only the provided evidence images. Use the listed Benchmark Director and
Normalized Genres for all calculations. Do not use outside knowledge.
Use population standard deviation whenever requested. Do not round intermediate
calculations or thresholds; round only the final values at the requested precision.
Do not guess if the evidence is insufficient.

Question:
{question}

Answer format:
{answer_format}

Return only the final answer.
''')
    write_json(out/'manifest.json',{'condition':'image-only normalized metadata','source_csv_sha256':sha(args.source_csv),'source_db_sha256':sha(args.source_db),'font_sha256':sha(args.font),'bold_font_sha256':sha(args.bold_font),'original_questions_sha256':sha(original_csv),'pdf_sha256':sha(pdf_path),'cards':records,'note':NOTE,'model_payload':'Only rendered PDF image data URLs plus question/instructions. No manifest or structured source is sent.'})
    print('Created 20 image-only cards, 20-page PDF, 45 questions and ground truths. No model calls.')

if __name__ == '__main__':
    main()
