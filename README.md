终端 1：导入服务，端口 8000
cd E:\code\agent\DocChat
conda activate DocChat
python -m knowledge.api.import_router
终端 2：问答服务，端口 8001
cd E:\code\agent\DocChat
conda activate DocChat
python -m knowledge.api.query_router
浏览器访问：
http://127.0.0.1:8000/front/import.html
http://127.0.0.1:8001/front/chat.html
