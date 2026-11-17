#!/usr/bin/env python3
"""Audit committed public release content without displaying credential material.

This checks the current tree (including frozen accelerated ZIP and notebooks);
it CANNOT scan historical git objects or prove that an exposed secret was not
already in public commit history. It never prints potential secret values.
"""
from __future__ import annotations
import json,re,sys,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TEXT_SUFFIX={".md",".py",".ipynb",".json",".yaml",".yml",".txt",".toml",".sh",".cff",".csv"}
HARD={
 "github_token":re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9_]{25,}|github_pat_[A-Za-z0-9_]{35,})\b"),
 "openai_key":re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{25,}\b"),
 "google_api_key":re.compile(rb"\bAIza[A-Za-z0-9_-]{30,}\b"),
 "pem_private_key":re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
 "aws_access_key":re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
 "password_assignment":re.compile(rb"(?im)^\s*(?:password|client_secret|api_key|access_token)\s*=\s*['\"][^'\"\r\n]{12,}['\"]"),
}
DISCLOSE={
 "external_drive_links":re.compile(rb"https?://drive\.google\.com/"),
 "public_email_addresses":re.compile(rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
 "absolute_home_paths":re.compile(rb"(?:/Users/|/home/[A-Za-z0-9_-]+/|C:\\Users\\)"),
 "notebook_embedded_outputs":re.compile(rb'"outputs"\s*:\s*\[\s*\{'),
}
def scan_bytes(path,raw):
    return {
      "path":path,"bytes":len(raw),
      "high_risk_patterns":[name for name,p in HARD.items() if p.search(raw)],
      "review_counts":{name:len(p.findall(raw)) for name,p in DISCLOSE.items() if p.search(raw)}
    }

def main():
    outcomes=[]
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or ".git" in p.parts or "__pycache__" in p.parts:
            continue
        rel=p.relative_to(ROOT).as_posix()
        if p.suffix.lower() in TEXT_SUFFIX or p.name in {".gitignore","LICENSE"}:
            if p.stat().st_size < 10_000_000:
                outcomes.append(scan_bytes(rel,p.read_bytes()))
        elif p.suffix==".zip":
            with zipfile.ZipFile(p) as archive:
                for member in archive.infolist():
                    if member.is_dir():continue
                    n=member.filename
                    if (member.file_size<=8_000_000 and
                        Path(n).suffix.lower() in TEXT_SUFFIX):
                        outcomes.append(scan_bytes(rel+"::"+n,archive.read(member)))
    findings=[x for x in outcomes if x["high_risk_patterns"]]
    review=[x for x in outcomes if x["review_counts"]]
    result={
      "files_and_archive_members_scanned":len(outcomes),
      "high_risk_match_count":len(findings),
      "high_risk_paths_and_pattern_types_only":[
         {"path":x["path"],"patterns":x["high_risk_patterns"]} for x in findings],
      "review_only_public_exposure_counts":[
         {"path":x["path"],**x["review_counts"]} for x in review],
      "does_not_scan_git_history":True,
      "is_not_comprehensive_third_party_secrets_or_license_audit":True
    }
    print("SANAD_PUBLIC_RELEASE_AUDIT="+json.dumps(result,sort_keys=True))
    if findings:
        print("SECURITY HOLD: do not publish more files until credential findings are resolved.",file=sys.stderr)
        raise SystemExit(1)

if __name__=="__main__":
    main()
