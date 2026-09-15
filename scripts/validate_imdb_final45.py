"""Offline independent arithmetic verification and artifact checks for final45."""
import argparse
import base64
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from decimal import Decimal as D, ROUND_HALF_UP
from pathlib import Path
from statistics import median
import pymupdf
import yaml
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from unified_pipeline.evidence.visual_pdf_evidence import VisualPdfEvidenceBuilder

def compute_answers(rows):
    rs=[dict(r,year=int(r['year']),runtime_minutes=D(r['runtime_minutes']),imdb_rating=D(r['imdb_rating']),genres=r['genres'].split('; ')) for r in rows]
    mean=lambda xs:sum(xs)/len(xs)
    fmt=lambda n:str(n.quantize(D('.01'),rounding=ROUND_HALF_UP))
    groups={g:[r for r in rs if g in r['genres']] for g in sorted({g for r in rs for g in r['genres']})}
    a={}; details={}
    def avg(sub,col):return mean([r[col] for r in sub])
    def out(n,sub,col,sign):
        vals=[r[col] for r in sub];m=mean(vals);sd=mean([(x-m)**2 for x in vals]).sqrt();t=m+sign*sd
        a[n]='; '.join(r['title'] for r in sub if (r[col]>t if sign==1 else r[col]<t))
        details[n]={'n':len(vals),'sum':str(sum(vals)),'mean':str(m),'population_sd':str(sd),'threshold':str(t)}
    before=[r for r in rs if r['year']<2010];after=[r for r in rs if r['year']>=2010]
    a[1]=fmt(avg(after,'imdb_rating'))
    a[4]=str(sum(r['runtime_minutes'] for r in groups['Adventure'] if r['year']>=2010))+' minutes'
    repeated=Counter(r['director'] for r in rs)
    a[6]=fmt(avg([r for r in rs if repeated[r['director']]>1],'runtime_minutes'))+' minutes'
    vals=sorted(r['imdb_rating'] for r in rs);q1=median(vals[:10]);q3=median(vals[10:]);iqr=q3-q1
    a[7]='; '.join(f"{r['title']} - {r['imdb_rating']}" for r in rs if r['imdb_rating']<q1-D('1.5')*iqr or r['imdb_rating']>q3+D('1.5')*iqr)
    m1=avg([r for r in rs if r['year']<2000],'imdb_rating');m2=avg([r for r in rs if r['year']>=2000],'imdb_rating')
    a[8]=f'Before 2000: {fmt(m1)}; 2000 or after: {fmt(m2)}; Difference: {fmt(m1-m2)}'
    a[10]=fmt(avg([r for r in rs if r['runtime_minutes']>avg(rs,'runtime_minutes')],'imdb_rating'))
    for n,pred,col,isavg in [(11,lambda r:r['year']<2000,'runtime_minutes',False),(12,lambda r:r['year']>=2010,'runtime_minutes',True),(13,lambda r:r['director']=='Christopher Nolan','imdb_rating',True),(14,lambda r:r['runtime_minutes']>=150,'imdb_rating',True),(15,lambda r:r['imdb_rating']>=D('8.5'),'runtime_minutes',True),(31,lambda r:r['imdb_rating']>=D('8.5'),'runtime_minutes',False),(33,lambda r:r['runtime_minutes']<120,'imdb_rating',True),(34,lambda r:D('8.0')<=r['imdb_rating']<=D('8.4'),'runtime_minutes',True),(35,lambda r:r['director']=='Christopher Nolan','runtime_minutes',False)]:
        sub=[r for r in rs if pred(r)];total=sum(r[col] for r in sub);m=total/len(sub)
        a[n]=(fmt(m) if isavg else str(total))+(' minutes' if col=='runtime_minutes' else '')
        details[n]={'n':len(sub),'sum':str(total),'mean':str(m)}
    m1=avg(before,'runtime_minutes');m2=avg(after,'runtime_minutes')
    a[32]=f'Before 2010: {fmt(m1)} minutes; 2010 or after: {fmt(m2)} minutes; Difference: {fmt(m1-m2)} minutes'
    metrics={
      2:(4,lambda x:avg(x,'imdb_rating'),min,fmt),
      5:(3,lambda x:max(r['imdb_rating'] for r in x)-min(r['imdb_rating'] for r in x),max,lambda v:f'{v} rating points'),
      9:(3,lambda x:sum(r['runtime_minutes'] for r in x),max,lambda v:f'{v} minutes'),
      16:(4,lambda x:avg(x,'imdb_rating'),max,fmt),
      17:(4,lambda x:sum(r['runtime_minutes'] for r in x),min,lambda v:f'{v} minutes'),
      18:(1,len,max,lambda v:f'{v} movies'),
      19:(4,lambda x:avg(x,'runtime_minutes'),max,lambda v:f'{fmt(v)} minutes'),
      20:(4,lambda x:avg(x,'runtime_minutes'),min,lambda v:f'{fmt(v)} minutes'),
      21:(4,lambda x:D(sum(r['year']>=2010 for r in x))*100/len(x),max,lambda v:f'{fmt(v)}%'),
      22:(4,lambda x:sum(r['imdb_rating']>=D('8.5') for r in x),max,lambda v:f'{v} movies'),
      36:(4,lambda x:max(r['runtime_minutes'] for r in x)-min(r['runtime_minutes'] for r in x),max,lambda v:f'{v} minutes'),
      37:(4,lambda x:D(sum(r['year'] for r in x))/len(x),max,fmt),
      38:(4,lambda x:sum(r['year']>=2010 for r in x),max,lambda v:f'{v} movies'),
      39:(4,lambda x:sum(r['runtime_minutes'] for r in x),max,lambda v:f'{v} minutes')}
    for n,(minimum,fn,choose,formatting) in metrics.items():
        candidates={g:fn(sub) for g,sub in groups.items() if (len(sub)==4 if n==39 else len(sub)>=minimum)}
        best=choose(candidates.values());winners=[g for g,v in candidates.items() if v==best];assert len(winners)==1,(n,winners)
        a[n]=f'{winners[0]} - {formatting(best)}'
        details[n]={'candidate_values':{k:str(v) for k,v in candidates.items()},'ties':False}
    m1=avg(groups['Drama'],'imdb_rating');m2=avg(groups['Adventure'],'imdb_rating')
    a[40]=f'Drama - {fmt(m1)}; Adventure - {fmt(m2)}; Difference: {fmt(m1-m2)}'
    for n,sub,col,sign in [(3,[r for r in rs if r['year']>=2000],'runtime_minutes',1),(23,rs,'runtime_minutes',1),(24,rs,'runtime_minutes',-1),(25,rs,'imdb_rating',1),(26,groups['Adventure'],'runtime_minutes',1),(27,groups['Adventure'],'runtime_minutes',-1),(28,groups['Drama'],'runtime_minutes',1),(29,groups['Action'],'runtime_minutes',1),(30,groups['Action'],'runtime_minutes',-1),(41,before,'runtime_minutes',1),(42,before,'runtime_minutes',-1),(43,after,'runtime_minutes',1),(44,after,'runtime_minutes',-1),(45,groups['Action'],'imdb_rating',1)]:out(n,sub,col,sign)
    assert set(a)==set(range(1,46))
    return a,details

def validate():
    out=ROOT/'evidence/imdb_20_final45'
    rows=list(csv.DictReader(io.StringIO((out/'frozen_movies.csv').read_text())))
    answers,details=compute_answers(rows)
    table=list(csv.DictReader(io.StringIO((out/'focused_45_question_pdf_questions.csv').read_text())))
    questions=json.loads((out/'questions.json').read_text())['questions']
    assert len(table)==len(questions)==45
    assert [q['id'] for q in questions]==[f'F{i}' for i in range(1,46)]
    assert questions[:10]==json.loads((ROOT/'evidence/imdb_20/questions.json').read_text())['questions']
    assert table[:10]==list(csv.DictReader(io.StringIO((ROOT/'evidence/imdb_20/focused_10_question_pdf_questions.csv').read_text())))
    for r,q in zip(table,questions):
        n=int(q['id'][1:]);assert r['expected_answer']==answers[n],(n,r['expected_answer'],answers[n])
        assert r['question']==q['text'] and r['category']==q['category'] and r['batch']==q['source_pdf']=='metadata'
        g=ROOT/'ground_truth/imdb_20_final45'/q['ground_truth_file']
        assert list(csv.DictReader(io.StringIO(g.read_text())))==[{'answer_text':answers[n]}]
        if n<=10:assert g.read_bytes()==(ROOT/'ground_truth/imdb_20'/g.name).read_bytes()
        if q['category']=='statistical_outlier_detection' and n>10:assert 'population standard deviation' in q['text']
    counts=dict(Counter(q['category'] for q in questions));assert list(counts.values())==[15,15,15]
    config=yaml.safe_load((ROOT/'datasets/imdb_20_final45.yaml').read_text());config['_source_pdf']='metadata'
    result=VisualPdfEvidenceBuilder().build('',config)
    assert config['page_layout']=='metadata_only' and config['pages_per_image']==4
    assert config['render_scale']==1.0 and config['jpeg_quality']==75
    assert result.text=='' and len(result.metadata['images'])==5
    manifest=json.loads((out/'manifest.json').read_text())
    assert hashlib.sha256((out/'frozen_movies.csv').read_bytes()).hexdigest()==manifest['source_csv_sha256']
    assert len({r['title'] for r in rows})==20
    with pymupdf.open(ROOT/config['evidence_path']) as document:
        assert len(document)==20 and document.embfile_count()==0
        assert len(rows)==len(manifest['cards'])==20
        rendered=[]
        for index,(page,r,record) in enumerate(zip(document,rows,manifest['cards'])):
            assert page.get_text()=='' and not list(page.annots() or [])
            assert len(page.get_images())==1
            assert record['page_index']==index and record['fields']=={k:r[k] for k in ['title','year','runtime_minutes','imdb_rating','director','genres']}
            original=Image.open(ROOT/record['card']).convert('RGB')
            pix=page.get_pixmap(alpha=False)
            assert original.tobytes()==pix.samples
            rendered.append(original)
        # Independently reconstruct each 2x2 payload, checking all four positions
        # and the exact JPEG bytes, not merely the number of returned images.
        composite_hashes=[]
        for i,url in enumerate(result.metadata['images']):
            expected=Image.new('RGB',(2420,1820),'white')
            for card,position in zip(rendered[i*4:i*4+4],[(0,0),(1220,0),(0,920),(1220,920)]):
                expected.paste(card,position)
            encoded=io.BytesIO();expected.save(encoded,format='JPEG',quality=75,optimize=True)
            actual=base64.b64decode(url.split(',',1)[1])
            assert actual==encoded.getvalue(),f'Composite {i+1} differs from frozen cards'
            assert Image.open(io.BytesIO(actual)).size==(2420,1820)
            composite_hashes.append(hashlib.sha256(actual).hexdigest())
    old=yaml.safe_load((ROOT/'datasets/imdb_20.yaml').read_text());old['_source_pdf']='metadata'
    historical=VisualPdfEvidenceBuilder().build('',old)
    assert len(historical.metadata['images'])==10
    prior=ROOT/'results/imdb_20_semantic_evaluation_session/rendered_evidence'
    if prior.exists():
        for i,url in enumerate(historical.metadata['images'],1):assert base64.b64decode(url.split(',',1)[1])==(prior/f'pair_{i:02}.jpg').read_bytes()
    return {'offline_checks':'PASS','questions_verified':45,'category_counts':counts,'cards_verified':20,'pdf_pages':20,'model_images':5,'pages_per_image':4,'model_image_size':[2420,1820],'composite_sha256':composite_hashes,'evidence_text_empty':True,'pdf_text_layer_absent':True,'historical_rendering_unchanged':True,'F1_F10_unchanged':True,'computations':{f'F{n}':{'answer':answers[n],**details.get(n,{})} for n in range(1,46)}}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args()
    result=validate()
    if args.report:
        with args.report.open('x') as f:json.dump(result,f,indent=2)
    print('PASS: 45 recomputed answers, 20 cards/pages, 5 composite model images, frozen F1-F10 and historical rendering.')
