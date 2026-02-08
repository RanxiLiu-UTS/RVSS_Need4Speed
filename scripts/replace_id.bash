for f in ../data/train_01_turning/*.jpg; do
  dir=$(dirname "$f")
  base=$(basename "$f")

  new="2${base:1}"

  mv "$f" "$dir/$new"
done