#quickstart script
#echo "Downloading Data"
#wget -O test_data http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz
#echo "Preparing Data"
#python prepare_data.py test_data prepared_data
echo "Learning net"
CUDA_VISIBLE_DEVICES=1 python3 han.py ../data/Food_Prepared_Data.pkl --cuda --emb ../data/Food_Prepared_Data_emb_wmc2 --save models/Food_wmc2.save --output models/Food_wmc2.output
