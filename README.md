# Transformer-based highlights extraction from scientific papers (THExt demo)

This repository contains a demo of the model proposed in the paper "Transformer-based highlights extraction from scientific papers". Each `md` file contains a set of highlights extracted from the papers accepted at ACL 2021 conference. Specifically:

- `ACL2021_THExt_AI.md` contains the output of the model fine-tuned to extract highlights from "Artificial Intelligence" categorized journal papers.
- `ACL2021_THExt_CS.md` contains the output of the model fine-tuned to extract highlights from "Computer Science" categorized journal papers.

Any manual pre- or post- processing has been applied for highlights extraction. The text of the papers has been parsed from PDF files using [GROBID](https://grobid.readthedocs.io/en/latest/).
