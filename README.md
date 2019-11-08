notebook的数据结构：
list//表示一个notebook
[
	dic{//表示一段代码
		“text”: string//一段代码的辅助文本
		“code”:list//一段代码的若干行代码
			    [
				list[//表示一个段
					dic{//表示一行代码
					“code”:string//一行代码的代码文本
					“primitive”: list[]//一行代码的primi
					“notes”: string//一行代码的辅助文本
				}, 
			     dic{},
			      …
				     ]
			   ]
	}
	dic{},
	…
]

在read_ipynb.py的最后一行修改要读入的文件。
运行classify_code.py打印初步结果。
get_code_workflow.py还没写完，正在修改。
辅助信息（小可）可以单独写一个文档，使用from classify_code import lis，lis就是notebook的数据结构。
工作流（子浩，雅馨）可以直接修改get_code_workflow.py。也可重写。
if else（钦亮，陆绿）等处理重建文档。
