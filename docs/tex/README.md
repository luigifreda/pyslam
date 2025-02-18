The pdf document is just a first draft structure to be completed. WIP.


To create a first sketch of the tex:
```bash
sudo apt install pandoc
pandoc --from=markdown --to=latex ../../README.md --output=document_new.tex --highlight-style=espresso --standalone
```