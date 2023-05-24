# Sync a clean repository to huggingface spaces
# Set HF_USER and HF_TOKEN in environment variables first

rm -rf spaces && mkdir spaces && cd spaces

cp ../bcgrun_web.py . && cp ../requirements.txt . && cp -r ../bcgunet .

cat ../spaces.yml ../README.md > README.md

git init -b main && git add . && git commit -m "Create Space"
git push https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/bcg-unet/demo.git main -f

cd .. && rm -rf spaces
