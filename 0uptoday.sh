#! /bin/sh

git add .

commit="$1"

if [ -z "$commit" ]
then
	git commit -m "commit"
	# echo "Please set \$commit"

else
	git commit -m "$1"
	# echo "Setting up jail at $commit"
fi

git push -u origin adelo
