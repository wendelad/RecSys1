using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;


/*  cria uma classe de pre processamento vai receber 3 parametros  , o arquivo principal , numero de kfolds e a porcentagem de
treino e test , vai gerar os arquivos com os k fold , depois vai ler dos k fold e gerar para cada k fold um training e test
com  a quantidade determinada pelo usuiario  , training e 80% o test e 20% (valor a ser definido pelo usuario), 
traning partial vai ser opcional com 20% a menos que o training  o test com os msm 20% do test  de forma aleatoria
gerar uma matriz de saida mostrando o arquivos gerados training e test e os k fold se possível*
ex se for gerado 10 arquivos , os arquivos 1-8 serão o training e o 9-10 serão o test é necessario fazer concatenar esses arquivos 
para ser gerados os training e test 
*/

// 

namespace WindowsFormsApplication1
{

    public partial class Form1 : Form
    {
        const string Arquivoteste = @"C:\Users\laique\Downloads\kfolds\Kfold-parte-1.txt";
        private string strPathFile = @"C:\Users\laique\Downloads\kfolds\Data_Default.txt";
        const string destinationFileName = @"C:\Users\laique\Downloads\kfolds\Kfold-parte-{0}.txt";
        const string destinationFileNamepasta = @"C:\Users\laique\Downloads\kfolds\Kfold-pasta-{0}";
        string diretorio = @"C:\Users\laique\Downloads\kfolds";
        public double contador3;
        public int linhasporarquivo = 0;
        public Boolean cb_marcado;
        private static int contador1;
        public int contadorporcento = 0;
        public int flag = 0;
        public Form1()
        {
            InitializeComponent();
        }



        private void btnAbrir_Click(object sender, EventArgs e)
        {
            Abrir();
        }

        private void Abrir()

        {

            try

            {

                //Verifico se o arquivo que desejo abrir existe e passo como parâmetro a respectiva variável

                if (File.Exists(strPathFile))

                {

                    //Se existir "starto" um processo do sistema para abrir o arquivo e, sem precisar

                    //passar ao processo o aplicativo a ser aberto, ele abre automaticamente o Notepad

                    System.Diagnostics.Process.Start(strPathFile);


                }

                else

                {

                    //Se não existir exibo a mensagem

                    MessageBox.Show("Arquivo não encontrado!");

                }

            }

            catch (Exception ex)

            {

                MessageBox.Show(ex.Message);

            }

        }

        private void btnConcatenar_Click(object sender, EventArgs e)
        {
            Concatenar();
        }
        private void Concatenar()

        {
            //string diretorio = @"C:\Users\Bruno\Downloads\kfolds";
            //string caminhoArquivoDestino = @"C:\Users\Bruno\Downloads\kfolds\Data_Default.txt";




            String[] listaDeArquivos = Directory.GetFiles(diretorio);

            if (listaDeArquivos.Length > 0)
            {


                FileStream arquivoDestino = File.Open(strPathFile, FileMode.OpenOrCreate);
                arquivoDestino.Close();

                List<String> linhasDestino = new List<string>();

                foreach (String caminhoArquivo in listaDeArquivos)
                {
                    linhasDestino.AddRange(File.ReadAllLines(caminhoArquivo));
                }

                File.WriteAllLines(strPathFile, linhasDestino.ToArray());
                MessageBox.Show("Arquivo atualizado!");
                ExcluirKfolds();
                int numero = 1;
                for (; numero <= 10; numero++)
                    File.Delete(@"C:\Users\laique\Downloads\kfolds\Kfold-parte-" + numero + ".txt");

            }
            //Exibo a mensagem que o arquivo foi atualizado                      

            else

            {
                //Se não existir exibo a mensagem

                MessageBox.Show("Arquivo não encontrado!");

            }

        }

        private void btnAlterar_Click(object sender, EventArgs e)
        {
            Alterar(10);
        }
        private void Alterar(int valorNum1)

        {

            try

            {
              //  int valorNum1 = int.Parse(textBox1.Text);
                linhasporarquivo = valorNum1;
                //Verifico se o arquivo que desejo abrir existe e passo como parâmetro a variável respectiva

                if (File.Exists(strPathFile))

                {

                    //Instancio o FileStream passando como parâmetro a variável padrão, o FileMode que será

                    //o modo Open e o FileAccess, que será Read(somente leitura). Este método é diferente dos

                    //demais: primeiro irei abrir o arquivo, depois criar um FileStream temporário que irá

                    //armazenar os novos dados e depois criarei outro FileStream para fazer a junção dos dois

                    using (FileStream fs = new FileStream(strPathFile, FileMode.Open, FileAccess.Read))

                    {

                        //Aqui instancio o StreamReader passando como parâmetro o FileStream criado acima.

                        //Uso o StreamReader já que faço 1º a leitura do arquivo. Irei percorrer o arquivo e

                        //quando encontrar uma string qualquer farei a alteração por outra string qualquer

                        using (StreamReader sr = new StreamReader(fs))

                        {

                            //Crio o FileStream temporário onde irei gravar as informações

                            using (FileStream fsTmp = new FileStream(strPathFile + ".tmp",

                                                       FileMode.Create, FileAccess.Write))

                            {

                                //Instancio o StreamWriter para escrever os dados no arquivo temporário,

                                //passando como parâmetro a variável fsTmp, referente ao FileStream criado

                                using (StreamWriter sw = new StreamWriter(fsTmp))

                                {


                                    // flag para garantir que so ira contar o numero de linhas do arquivo uma vez garantindo que  
                                    //a % seja em cima da quantidade original de linhas
                                    if (flag != 1)
                                    {
                                        int contador = 0;


                                        string strlinha = null;
                                        // enquanto houver linhas no arquivo vai incrementando o contador ao final vai ter a quantidade total de linhas
                                        while ((strlinha = sr.ReadLine()) != null)

                                        {
                                            contador++;

                                        }
                                        flag = 1;
                                        contador1 = contador;

                                    }

                                    // pega o valor do textbox convertendo para inteiro textbox1 é a quantidade de k folds
                                    

                                    linhasporarquivo = contador1 / linhasporarquivo;

                                    //Instancio o StreamWriter para escrever os dados no arquivo 
                                    // codigo para geração dos k folds pecorre o arquivo original escrevendo os dados em um outro arquivo

                                    using (var sourcefile = new StreamReader(strPathFile))
                                    {

                                        var fileCounter = 0;
                                        var destinationFile = new StreamWriter(string.Format(destinationFileName, fileCounter + 1));

                                        try
                                        {
                                            var lineCounter = 0;
                                            string line;

                                            // reseto o apontador para o inicio do arquivo
                                            fs.Seek(0, SeekOrigin.Begin);

                                            while ((line = sr.ReadLine()) != null)
                                            {

                                                // linhas por arquivo é a calculo feito quantas linhas vai ter em cada k fold de acordo com a % escolhida 
                                                // quando esse limite for antigido ira mudar para  o proximo arquivo . ficando asism kfold-parte-1 , kfold-parte-2 ...
                                                if (lineCounter >= linhasporarquivo)
                                                {
                                                    //sim.. hora de mudar de arquivo
                                                    lineCounter = 0;
                                                    fileCounter++;
                                                    destinationFile.Dispose();

                                                    destinationFile = new StreamWriter(string.Format(destinationFileName, fileCounter + 1));

                                                }
                                                destinationFile.WriteLine(line);
                                                lineCounter++;


                                            }
                                        }
                                        finally
                                        {
                                            destinationFile.Dispose();
                                        }
                                    }




                                }
                            }
                        }
                    }




                    //Ao final excluo o arquivo anterior e movo o temporário no lugar do original

                    //Dessa forma não perco os dados de modificação de meu arquivo

                    File.Delete(strPathFile);



                    //No método Move passo o arquivo de origem, o temporário, e o de destino, o original

                    File.Move(strPathFile + ".tmp", strPathFile);



                    //Exibo a mensagem ao usuário

                    MessageBox.Show("Arquivo alterado com sucesso!");
                   Excluir();

                }

                else

                {

                    //Se não existir exibo a mensagem

                    MessageBox.Show("Arquivo não encontrado!");

                }

            }

            catch (Exception ex)

            {

                MessageBox.Show(" O campo de quantidade de kfolds está vazio  ");

            }

        }

        
        private void Excluir()

        {
            try
            {
                //Verifico se o arquivo que desejo abrir existe e passo como parâmetro a variável respectiva          
                if (File.Exists(strPathFile))

                {
                    //Se existir chamo o método Delete da classe File para apagá-lo e exibo uma msg ao usuário
                    File.Delete(strPathFile);
                }

            }

            catch (Exception ex)

            {
                MessageBox.Show(ex.Message);
            }

        }

        private void ExcluirKfolds()

        {

            String[] listaDeArquivos = Directory.GetFiles(diretorio);
            try

            {

                //Verifico se o arquivo que desejo abrir existe e passo como parâmetro a variável respectiva

                if (File.Exists(destinationFileName))

                {

                    //Se existir chamo o método Delete da classe File para apagá-lo e exibo uma msg ao usuário

                    File.Delete(destinationFileName);

                }

            }

            catch (Exception ex)

            {

                MessageBox.Show(ex.Message);

            }

        }


        private void button1_Click_1(object sender, EventArgs e)
        {

        }

        private void button2_Click(object sender, EventArgs e)
        {

        }


        private void richTextBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            //int valorNum1 = int.Parse(textBox1.Text);
            //linhasporarquivo = valorNum1;
        }

        private void button1_Click_2(object sender, EventArgs e)
        {
            gerarTestTrainingSet(10,20);
        }
        private void gerarTestTrainingSet(int tam, double tamTestSet)
        {


            if (File.Exists(Arquivoteste))

            {

                // configuração das variaveis onde vão ser lidas os arquivos...
                string nomeDaPasta = @"C:\Users\laique\Downloads\kfolds\kfold-pasta-";
                string caminhoArquivoDestinoTest = @"C:\Users\laique\Downloads\kfolds\test.txt";
                string caminhoArquivoDestinotreino = @"C:\Users\laique\Downloads\kfolds\training.txt";
                DirectoryInfo dir = new DirectoryInfo(@"C:\Users\laique\Downloads\kfolds");


                // faz as leituras dos dados inseridos na interface fazendo as devidas conversões de valores



                //double tam = double.Parse(textBox2.Text);
                //  double tamTestSet = double.Parse(textBox3.Text);
                double tamTrainingSet = tam * (tamTestSet / 100);
                tamTrainingSet = Math.Round(tamTrainingSet);
                double percentTrainingSet = tam - tamTrainingSet;

                var fileCounter = 0;

                HashSet<int> totalSet = new HashSet<int>();

                // preenche o total set e cria as pasta dos k folds
                for (int i = 0; i < tam; i++)
                {
                    totalSet.Add(i);

                    var destino = Directory.CreateDirectory(string.Format(destinationFileNamepasta, fileCounter + 1));

                    fileCounter++;
                }

                Random random = new Random();
                HashSet<int> testSet = new HashSet<int>();

                // estas listas serão usadas para salvar os valores originais 
                List<int> teste = new List<int>();
                List<int> treino = new List<int>();


                // loop sobre a quantidade de arquivos para gerar aleatoriamente os test e training  e guardar em uma lista
                int num = 1;
                for (int i = 0; i < tam; i++, num++)
                {
                    do
                    {
                        int index = totalSet.ToList()[random.Next(0, totalSet.Count)];
                        testSet.Add(index);
                        totalSet.Remove(index);
                    } while (testSet.Count != tamTrainingSet);



                    Console.Write(" agora o training ");
                    foreach (int training in totalSet)
                    {

                        Console.Write(" " + training);
                    }
                    treino = totalSet.ToList();

                    Console.Write(" agora o test ");
                    foreach (int test in testSet)
                    {

                        Console.Write(" " + test);

                        totalSet.Add(test);
                    }
                    teste = testSet.ToList();
                    testSet.Clear();



                    // gerar test  

                    // pego todos os arquivos contidos na pasta "diretorio" que é uma variavel declarada no inicio com caminho da pasta
                    String[] listaDeArquivos = Directory.GetFiles(diretorio);

                    // se possuir arquivos faz
                    if (listaDeArquivos.Length > 0)
                    {

                        int k = 0;

                        // cria o arquivo de test
                        FileStream arquivoDestino = File.Open(caminhoArquivoDestinoTest, FileMode.OpenOrCreate);
                        arquivoDestino.Close();

                        List<String> linhasDestino = new List<string>();
                        int arquivoselecionado;


                        //fileCounter = 0;

                        for (k = 0; k < teste.Count; k++)
                        {


                            arquivoselecionado = teste[k];


                            linhasDestino.AddRange(File.ReadAllLines(listaDeArquivos[arquivoselecionado]));


                            File.WriteAllLines(caminhoArquivoDestinoTest, linhasDestino.ToArray());

                        }
                        string destinoTest = nomeDaPasta + num + "\\test.txt";
                        foreach (FileInfo f in dir.GetFiles("test.txt"))
                            File.Move(caminhoArquivoDestinoTest, destinoTest);

                    }



                    // gerar treino
                    String[] listaDeArquivostreino = Directory.GetFiles(diretorio);

                    if (listaDeArquivostreino.Length > 0)
                    {

                        int k = 0;


                        FileStream arquivoDestino = File.Open(caminhoArquivoDestinotreino, FileMode.OpenOrCreate);
                        arquivoDestino.Close();

                        List<String> linhasDestino = new List<string>();
                        int arquivoselecionado;

                        //var fileCounter = 0;

                        // pecorrer a lista de training
                        for (k = 0; k < treino.Count; k++)
                        {

                            // pego o arquivo na 1 posição da lista de training ler as linhas desse arquivo e escreve no arquivo de training no caminho destino
                            arquivoselecionado = treino[k];


                            linhasDestino.AddRange(File.ReadAllLines(listaDeArquivostreino[arquivoselecionado]));


                            File.WriteAllLines(caminhoArquivoDestinotreino, linhasDestino.ToArray());


                        }
                        string destinoTraining = nomeDaPasta + num + "\\training.txt";
                        foreach (FileInfo f in dir.GetFiles("training.txt"))
                            File.Move(caminhoArquivoDestinotreino, destinoTraining);

                    }

                }



            }
            else
                Console.Write("Arquivos kfolds não existem");
        }
        
        private void textBox2_TextChanged(object sender, EventArgs e)
        {


        }

        private void textBox3_TextChanged(object sender, EventArgs e)
        {

        }
    }
}


