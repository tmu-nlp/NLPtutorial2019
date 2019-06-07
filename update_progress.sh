while :
do
    git pull > mes
    date '+%R' > time
    if (! diff mes mesdef) || (diff time timedef); then
        python3 make_progress.py
        git add progress.png
        git commit -m "Progress Graph Auto-Update"
        git push
    fi
    rm mes
    rm time
    sleep 10m
done
