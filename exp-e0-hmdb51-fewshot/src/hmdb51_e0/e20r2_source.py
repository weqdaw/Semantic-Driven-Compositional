"""Verify historical E20R with its pinned snapshot without changing old evidence."""
from pathlib import Path
import subprocess,sys,json
from .e11_workflow import tree_hashes
from .provenance import read_json,sha256_file
from .e1_experiment import OFFICIAL

_EXPORT='''
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
from hmdb51_e0.e20r_workflow import load_packet
from hmdb51_e0.e11_workflow import tree_hashes
c=load_packet(Path(sys.argv[2]),require_official=True)
s=c['source']
print(json.dumps(dict(parent=str(c['packet']),parent_hashes=tree_hashes(c['packet']),mapping=c['mapping'],
 concepts=c['concepts'],text_embeddings=s['text_embeddings'].tolist(),encoder=s['encoder'],decoder=s['decoder'],
 selected_videos=c['plan']['selected_videos'],feature_condition=s['feature_condition'],
 protocol_id=s['protocol_id'],dataset_root=c['plan']['dataset_root'],plan_path=c['manifest']['plan_path'],
 plan_sha256=c['manifest']['plan_sha256'],e11_batch=str(s['e11_batch']),audit_directory=str(s['audit_directory']),
 batch_hashes=s['batch_hashes'],audit_hashes=s['audit_hashes']),ensure_ascii=False))
'''

def parent_context(parent):
    parent=Path(parent).resolve();manifest=read_json(parent/'input_manifest.json')
    snapshot=parent/'private/code_snapshot'
    if manifest.get('schema')!='e20r_packet_v1' or manifest.get('status')!='success':raise ValueError('complete E20R parent required')
    if manifest.get('feature_condition')!=OFFICIAL:raise ValueError('official frozen CLIP E20R parent required')
    if tree_hashes(snapshot)!=manifest['source_code']:raise ValueError('historical source snapshot differs from pinned hashes; use -B')
    # Execute only the fully hash-verified local snapshot in an isolated interpreter.
    result=subprocess.run([sys.executable,'-I','-B','-X','utf8','-c',_EXPORT,str(snapshot),str(parent)],
        text=True,encoding='utf-8',capture_output=True,check=False)
    if result.returncode:raise ValueError('historical E20R verification failed: '+result.stderr[-1500:])
    value=json.loads(result.stdout);verify_parent(value)
    return value

def verify_parent(value):
    if value['feature_condition']!=OFFICIAL:raise ValueError('official CLIP source required')
    if tree_hashes(value['parent'])!=value['parent_hashes']:raise ValueError('E20R parent changed')
    if sha256_file(value['plan_path'])!=value['plan_sha256']:raise ValueError('E20R source plan changed')
    if tree_hashes(value['e11_batch'])!=value['batch_hashes'] or tree_hashes(value['audit_directory'])!=value['audit_hashes']:
        raise ValueError('E20R ancestor evidence changed')
