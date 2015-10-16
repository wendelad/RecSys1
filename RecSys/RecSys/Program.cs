using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyMediaLite;
using MyMediaLite.Data;
using MyMediaLite.DataType;
using MyMediaLite.Eval;
using MyMediaLite.IO;
using MyMediaLite.ItemRecommendation;
using Mono.Options;
using MyMediaLite.Eval.Measures;
using btl.generic;
using GAF;
using GAF.Operators;


namespace RecSys
{
    class Program
    {
        //Recommender
        protected List<IRecommender> recommenders = new List<IRecommender>();
        protected string recommender_options = string.Empty;

        // data
        List<IPosOnlyFeedback> training_data = new List<IPosOnlyFeedback>();
        List<IPosOnlyFeedback> test_data = new List<IPosOnlyFeedback>();

        List<IPosOnlyFeedback> training_probe_data = new List<IPosOnlyFeedback>();
        List<IPosOnlyFeedback> test_probe_data = new List<IPosOnlyFeedback>();


        List<IList<int>> test_users = new List<IList<int>>();



        // ID mapping objects
        protected List<IMapping> user_mapping = new List<IMapping>();
        protected List<IMapping> item_mapping = new List<IMapping>();

        // command line parameters
        protected string training_file;
        protected string test_file;
        protected string training_partial_file;
        protected string test_partial_file;
        // help
        protected bool show_help = false;


        protected OptionSet options;

        //Eval
        protected IList<string> eval_measures;
        protected string measures;



        protected string arquivo;




        protected string item_attributes_file;
        protected List<IBooleanMatrix> item_attributes = new List<IBooleanMatrix>();




        //
        string[] arquivos = new string[] { "genres", "tags", "directors", "actors", "countries" };
        //string[] arquivos = new string[] { "genres", "tags", "directors", "countries" };
        //string[] arquivos = new string[] { "genres", "countries"  };




        protected void Usage(int exit_code)
        {

            Console.WriteLine(@"  method ARGUMENTS have the form name=value
general OPTIONS:
 --recommender=METHOD             use METHOD for recommendations (default: MostPopular) 
 --help                           display this usage information and exit




 --k-fold=Numero 

");
            Environment.Exit(exit_code);
        }

        static void Main(string[] args)
        {
            var program = new Program();
            program.Run(args);
        }


        protected void Run(string[] args)
        {
            Console.WriteLine("WISER-RecSys começou");

            options = new OptionSet() {
                // string-valued options
                 { "arquivo=",            v              => arquivo             = v },
                { "measures=",            v              => measures             = v },
                { "recommender-options=", v              => recommender_options += " " + v },
                { "help",                 v => show_help         = v != null },

            };

            eval_measures = ItemRecommendationEvaluationResults.DefaultMeasuresToShow;

            IList<string> extra_args = options.Parse(args);

            if (show_help)
                Usage(0);


            //eval
            if (measures != null)
                eval_measures = measures.Split(' ', ',');





            //Rodar o de vocs



            // 
            training_file = "training.data";
            test_file = "test.data";
            training_partial_file = "training.partial.data";
            test_partial_file = "test.partial.data";



            for (int i = 0; i < arquivos.Length; i++)
            {

                MyMediaLite.Random.Seed = 1;


                item_attributes_file = "movie_" + arquivos[i] + ".dat_saida";


                user_mapping.Add(new Mapping());
                item_mapping.Add(new Mapping());



                //Setup recommender
                recommenders.Add("BPRMFAttr".CreateItemRecommender());
                recommenders[i].Configure(recommender_options, (string msg) =>
                {
                    Console.Error.WriteLine(msg); Environment.Exit(-1);
                });


                // item attributes
                if (recommenders[i] is IItemAttributeAwareRecommender && item_attributes_file == null)
                    Abort("Recommender expects --item-attributes=FILE.");


                if (item_attributes_file != null)
                    item_attributes.Add(AttributeData.Read(item_attributes_file, item_mapping[i]));
                if (recommenders[i] is IItemAttributeAwareRecommender)
                    ((IItemAttributeAwareRecommender)recommenders[i]).ItemAttributes = item_attributes[i];


                IBooleanMatrix lista_vazia = new SparseBooleanMatrix();
                if (recommenders[i] is IUserAttributeAwareRecommender)
                    ((IUserAttributeAwareRecommender)recommenders[i]).UserAttributes = lista_vazia;


                // training data
                training_data.Add(ItemData.Read(training_file, user_mapping[i], item_mapping[i], false));

                test_data.Add(ItemData.Read(test_file, user_mapping[i], item_mapping[i], false));


                test_users.Add(test_data[i].AllUsers);


                //Probe

                training_probe_data.Add(ItemData.Read(training_partial_file, user_mapping[i], item_mapping[i], false));
                test_probe_data.Add(ItemData.Read(test_partial_file, user_mapping[i], item_mapping[i], false));


                if (recommenders[i] is MyMediaLite.ItemRecommendation.ItemRecommender)
                    ((ItemRecommender)recommenders[i]).Feedback = training_probe_data[i];


                //Trainar
                Console.WriteLine("Vamos ao probe training");
                var train_time_span = Wrap.MeasureTime(delegate () { recommenders[i].Train(); });
                Console.WriteLine("training_time " + train_time_span + " ");


            }

            Evaluation evaluation = new Evaluation(recommenders, test_probe_data, training_probe_data);

            //Probe learn
            Console.WriteLine("Probe learn started");
            TimeSpan time_span = Wrap.MeasureTime(delegate () { evaluation.EvaluateProbe(test_users, user_mapping, item_mapping); });
            Console.WriteLine(" Probe learn time: " + time_span);


            for (int i = 0; i < arquivos.Length; i++)
            {

                MyMediaLite.Random.Seed = 1;


                item_attributes_file = "movie_" + arquivos[i] + ".dat_saida";


                //Setup recommender
                recommenders[i] = "BPRMFAttr".CreateItemRecommender();
                recommenders[i].Configure(recommender_options, (string msg) => { Console.Error.WriteLine(msg); Environment.Exit(-1); });


                // item attributes
                if (recommenders[i] is IItemAttributeAwareRecommender && item_attributes_file == null)
                    Abort("Recommender expects --item-attributes=FILE.");


                if (recommenders[i] is IItemAttributeAwareRecommender)
                    ((IItemAttributeAwareRecommender)recommenders[i]).ItemAttributes = item_attributes[i];


                IBooleanMatrix lista_vazia = new SparseBooleanMatrix();
                if (recommenders[i] is IUserAttributeAwareRecommender)
                    ((IUserAttributeAwareRecommender)recommenders[i]).UserAttributes = lista_vazia;


                if (recommenders[i] is MyMediaLite.ItemRecommendation.ItemRecommender)
                    ((ItemRecommender)recommenders[i]).Feedback = training_data[i];



                //Trainar
                Console.WriteLine("Agora ao treino normal");
                var train_time_span = Wrap.MeasureTime(delegate () { recommenders[i].Train(); });
                Console.WriteLine("training_time " + train_time_span + " ");

            }



            var results = evaluation.Evaluate(test_data, training_data, test_users, user_mapping, item_mapping);

            foreach (EvaluationResults result in results)
            {
                Console.WriteLine(result.ToString());
            }

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();

        }


        protected void Abort(string message)
        {
            Console.Error.WriteLine(message);
            Environment.Exit(-1);
        }


    }
}