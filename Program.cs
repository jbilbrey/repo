using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using Microsoft.ML.Transforms;
using System.Linq;

namespace TestDeleteMe
{
    class Program
    {
       
        static string dataset = Path.Combine(Directory.GetCurrentDirectory(), "mpg.txt");        
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "model");

        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            Console.WriteLine("The path to the model is... " + _modelPath.ToString());  // debug only
            //load tensorflow model
            var tensorFlowModel = mlContext.Model.LoadTensorFlowModel(_modelPath);
            
            var schema = tensorFlowModel.GetModelSchema();
            var inputSchema = tensorFlowModel.GetInputSchema();
                               
            Console.WriteLine("The data is..." + inputSchema.ToString());

            var reader = mlContext.Data.CreateTextLoader(new[] {
                new TextLoader.Column("dense_input_1", DataKind.Single, new[] {new TextLoader.Range(1,9)}),          
            }, separatorChar: '\t', hasHeader: true);

            // read the data
            var data = reader.Load(dataset);

            // print data to screen
            var inputs = mlContext.Data.CreateEnumerable<InputData>(data, reuseRowObject: false).ToArray();

            // print the data to the console
            for (int i = 0; i < inputs.Length; i++)
            {
                //var predictedLabel = engine.Predict(inputs[i]);

                for (int j = 0; j < inputs[i].Features.Length; j++)
                {
                    Console.Write(inputs[i].Features[j]);
                    Console.Write(" ");
                }
                //Console.WriteLine(predictedLabel.Output[0]);
            }

            ////////////// fit the model /////////////////// NEED HELP HERE to consume/use the model!!!

            ///var estimator = tensorFlowModel.ScoreTensorFlowModel("Predict", "dense_input_1").Fit(data);

        }
    }

    class InputData
    {
        [ColumnName("dense_input_1"), VectorType(9)]
        public float[]? Features { get; set; }
    }


    class OutputData
    {
        [ColumnName("Output"), VectorType(1)]
        public float[]? Prediction { get; set; }
    }
}
