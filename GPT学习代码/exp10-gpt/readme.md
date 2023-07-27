
## exp-10-gpt
  ## 1.tutorial
    指南类文件：
        （1）：以情感分析任务为例，介绍pipeline()的用法。
        （2）：介绍在pipeline()函数中如何使用另一个模型和标记器。
        （3）：Hugging Face Transformers库中AutoClass和AutoTokenizer的简要介绍
        （4）：Hugging Face Transformers库中的AutoModel和AutoModelForSequenceClassification的简单解释
        （5）：关于如何自定义构建模型的简单介绍。
        （6）：如何使用Hugging Face Transformers构建自定义模型
  ## 2.pipelines
    教程类文件：
        （1）：使用pipeline()进行推理。
        （2）：使用特定的tokenizer或model。
        （3）：使用pipeline()进行音频、视觉和多模态任务。
  ## 3.data-prepare
    教程类文件：
        （1）：对于文本数据，使用Tokenizer（分词器）将文本转换为令牌序列，创建令牌的数字表示，并将它们组装成张量。
        （2）：图像输入使用ImageProcessor（图像处理器）将图像转换为张量。
        （3）：多模态输入，使用Processor（处理器）将分词器和特征提取器或图像处理器组合在一起。
        （4）：对于文本，使用Tokenizer将文本转换为一系列的tokens，创建tokens的数字表示，并将它们组装成张量。
        （5）：对于图像输入，使用ImageProcessor将图像转换为张量。
        （6）：对于多模态输入，使用Processor结合tokenizer和feature extractor或image processor。
  ## 4.fintune
    教程类文件：
        （1）：如何对预训练模型进行微调
  ## 5.fintune-torch-only
    教程类文件：
        （1）：在原生PyTorch中进行微调训练
  ## 6.Distributed—training
    教程类文件：
        （1）：使用Accelerate进行分布式训练
        （2）：如何自定义原生PyTorch训练循环以在分布式环境中进行训练。
  ## 7.GPT
    介绍类文件：
        （1）：关于gpt模型的介绍
  ## 8.pretrain-gpt
    教程类文件：
        （1）：关于gpt语言模型微调的教程
  ## 9.bert
    介绍类文件：
        （1）：关于bert模型介绍
  ## 10.t5
    介绍类文件：
        （1）：关于t5模型介绍
  ## distributed-training
    代码类文件：
        （1）：使用Accelerator进行分布式训练的BERT文本分类模型的代码
  ## install
    必要库的安装