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
using MyMediaLite.Eval.Measures;
using GAF;
using GAF.Operators;

namespace RecSys
{
    class Ensemble
    {
        //List of prediction probes
        public List<IList<Tuple<int, float>>> list_prediction_probes = new List<IList<Tuple<int, float>>>();
        public HashSet<int> correct_items_global = new HashSet<int>();

        //Used in bestofall ensemble
        public Dictionary<string, int> best_alg = new Dictionary<string, int>();

        //Used in GA ensemble
        public Dictionary<string, List<double>> ga_weights = new Dictionary<string, List<double>>();


        //List of recommenders
        protected List<IRecommender> m_recommenders;

        //constructor 
        public Ensemble(List<IRecommender> recommenders)
        {
            m_recommenders = recommenders;
        }

        public List<KeyValuePair<int, float>> Ensenble(List<IList<Tuple<int, float>>> list)
        {
            var novaList = new Dictionary<int, float>();

            //Vamos criar uma nova lista com o score junto
            foreach (IList<Tuple<int, float>> aLista in list)
            {
                foreach (Tuple<int, float> item in aLista)
                {

                    if (novaList.ContainsKey(item.Item1))
                    {
                        if (novaList[item.Item1] < item.Item2)
                        {
                            novaList[item.Item1] = item.Item2;
                        }
                    }
                    else
                    {
                        novaList.Add(item.Item1, item.Item2);
                    }

                }



            }

            List<KeyValuePair<int, float>> listRetorno = novaList.ToList();

            listRetorno.Sort((firstPair, nextPair) =>
            {
                return nextPair.Value.CompareTo(firstPair.Value);
            });


            return listRetorno;



        }

        //Usado no 
        public List<KeyValuePair<int, float>> EnsenblePeso(double[] pesos)
        {
            var novaList = new Dictionary<int, float>();

            //Vamos criar uma nova lista com o score junto
            for (int i = 0; i < list_prediction_probes.Count; i++)
            {
                foreach (Tuple<int, float> item in list_prediction_probes[i])
                {
                    float value;
                    if (novaList.TryGetValue(item.Item1, out value))
                    {
                        novaList[item.Item1] = value + (item.Item2 * (float)pesos[i]);

                    }
                    else
                    {
                        novaList.Add(item.Item1, item.Item2 * (float)pesos[i]);
                    }

                }



            }

            List<KeyValuePair<int, float>> listRetorno = novaList.ToList();

            listRetorno.Sort((firstPair, nextPair) =>
            {
                return nextPair.Value.CompareTo(firstPair.Value);
            });


            return listRetorno;



        }



        public static bool Terminate(Population population, int currentGeneration, long currentEvaluation)
        {
            return currentGeneration > 70;
        }

        private double CalculateFitness(Chromosome chromosome)
        {


            //get x and y from the solution

            double[] values = new double[list_prediction_probes.Count];
            double rangeConst = 1 / (System.Math.Pow(2, 10) - 1);

            for (int i = 0; i < list_prediction_probes.Count; i++)
            {
                string str = chromosome.ToBinaryString((i * 10), 10);
                Int64 convertInt32 = Convert.ToInt32(str, 2);

                double x = (convertInt32 * rangeConst);

                values[i] = x;
            }




            var result = EnsenblePeso(values);

            var prediction_ensemble_probe = (from t in result select t.Key).ToArray();

            double resultado_ensemble = PrecisionAndRecall.AP(prediction_ensemble_probe, correct_items_global);

            return resultado_ensemble;


        }



        public void EvaluateProbe(List<IPosOnlyFeedback> test_probe_data, List<IPosOnlyFeedback> training_probe_data, List<IList<int>> test_users, List<IMapping> user_mapping,
            List<IMapping> item_mapping,
         int n = -1)
        {
            List<IList<int>> candidate_items = new List<IList<int>>();
            List<RepeatedEvents> repeated_events = new List<RepeatedEvents>();
            List<IBooleanMatrix> training_user_matrix = new List<IBooleanMatrix>();
            List<IBooleanMatrix> test_user_matrix = new List<IBooleanMatrix>();



            for (int i = 0; i < m_recommenders.Count; i++)
            {

                candidate_items.Add(new List<int>(test_probe_data[i].AllItems.Union(training_probe_data[i].AllItems)));
                repeated_events.Add(RepeatedEvents.No);


                if (candidate_items[i] == null)
                    throw new ArgumentNullException("candidate_items");
                if (test_probe_data[i] == null)
                    test_users[i] = test_probe_data[i].AllUsers;

                training_user_matrix.Add(training_probe_data[i].UserMatrix);
                test_user_matrix.Add(test_probe_data[i].UserMatrix);
            }
            int num_users = 0;
            var result = new ItemRecommendationEvaluationResults();

            // make sure that the user matrix is completely initialized before entering parallel code






            foreach (int user_id in test_users[0])
            {

                string original = user_mapping[0].ToOriginalID(user_id);


                List<IList<Tuple<int, float>>> list_of_predictions = new List<IList<Tuple<int, float>>>();

                HashSet<int> correct_items = new HashSet<int>();

                List<HashSet<int>> ignore_items_for_this_user = new List<HashSet<int>>();

                List<int> num_candidates_for_this_user = new List<int>();


                correct_items = new HashSet<int>(test_user_matrix[0][user_id]);
                correct_items.IntersectWith(candidate_items[0]);


                for (int i = 0; i < m_recommenders.Count; i++)
                {

                    int internalId = user_mapping[i].ToInternalID(original);


                    ignore_items_for_this_user.Add(new HashSet<int>(training_user_matrix[i][internalId]));



                    /* if (correct_items[i].Count == 0)
                         continue;
                     */

                    ignore_items_for_this_user[i].IntersectWith(candidate_items[i]);
                    num_candidates_for_this_user.Add(candidate_items[i].Count - ignore_items_for_this_user[i].Count);
                    /*if (correct_items[i].Count == num_candidates_for_this_user[i])
                        continue;
                    */


                    //Recomenda


                    var listaRecomendacao = m_recommenders[i].Recommend(user_id, candidate_items: candidate_items[i], n: n, ignore_items: ignore_items_for_this_user[i]);
                    for (int j = 0; j < listaRecomendacao.Count; j++)
                    {
                        string idOriginal = item_mapping[i].ToOriginalID(listaRecomendacao[j].Item1);
                        int idMappingZero = item_mapping[0].ToInternalID(idOriginal);


                        Tuple<int, float> tupla = new Tuple<int, float>(idMappingZero, listaRecomendacao[j].Item2);

                        listaRecomendacao[j] = tupla;
                    }

                    list_of_predictions.Add(listaRecomendacao);


                }



                //Usar o melhor
                double maiorMapping = 0;
                int idMaiorMapping = 0;

                //Testar cada individual
                for (int k = 0; k < list_of_predictions.Count; k++)
                {
                    int[] prediction_probe = (from t in list_of_predictions[k] select t.Item1).ToArray();


                    double resultado = PrecisionAndRecall.AP(prediction_probe, correct_items);

                    if (resultado > maiorMapping)
                    {
                        maiorMapping = resultado;
                        idMaiorMapping = k;

                    }


                }

                //Set global so Fitness itens can see.
                list_prediction_probes = list_of_predictions;
                correct_items_global = correct_items;

                //Algortimo Genetico
                /*   //  Crossover		= 80%
                   //  Mutation		=  5%
                   //  Population size = 100
                   //  Generations		= 2000
                   //  Genome size		= 2
                   GA ga = new GA(0.8, 0.05, 40, 400, list_prediction_probes.Count);

                   ga.FitnessFunction = new GAFunction(Fitness);

                   //ga.FitnessFile = @"H:\fitness.csv";
                   ga.Elitism = true;
                   ga.Go();

                   double[] values;
                   double fitness;
                   ga.GetBest(out values, out fitness);*/

                //create the GA using an initialised population and user defined Fitness Function 
                const double crossoverProbability = 0.85;
                const double mutationProbability = 0.08;
                const int elitismPercentage = 5;

                //create a Population of random chromosomes of length 44 
                var population = new Population(40, list_of_predictions.Count * 10, false, false);

                //create the genetic operators 
                var elite = new Elite(elitismPercentage);
                var crossover = new Crossover(crossoverProbability, true)
                {
                    CrossoverType = CrossoverType.DoublePoint
                };
                var mutation = new BinaryMutate(mutationProbability, true);

                //create the GA itself 
                var ga = new GeneticAlgorithm(population, CalculateFitness);

                //add the operators to the ga process pipeline 
                ga.Operators.Add(elite);
                ga.Operators.Add(crossover);
                ga.Operators.Add(mutation);

                //run the GA 
                ga.Run(Terminate);


                var best = population.GetTop(1)[0];
                double rangeConst = 1 / (System.Math.Pow(2, 10) - 1);
                ga_weights[original] = new List<double>();

                for (int i = 0; i < list_prediction_probes.Count; i++)
                {
                    string str = best.ToBinaryString((i * 10), 10);
                    Int64 convertInt32 = Convert.ToInt32(str, 2);

                    double x = (convertInt32 * rangeConst);

                    ga_weights[original].Add(x);
                }


                best_alg[original] = idMaiorMapping;
                num_users++;


                if (num_users % 10 == 0)
                    Console.Error.Write(".");
                if (num_users % 100 == 0)
                    Console.Error.WriteLine("");


            }


        }



    }
}
