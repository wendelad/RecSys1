// Copyright (C) 2012 Zeno Gantner
//
// This file is part of MyMediaLite.
//
// MyMediaLite is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// MyMediaLite is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with MyMediaLite.  If not, see <http://www.gnu.org/licenses/>.
//
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using MyMediaLite;
using MyMediaLite.DataType;
using MyMediaLite.Eval;
using MyMediaLite.IO;

namespace MyMediaLite.ItemRecommendation
{
	/// <summary>
	/// BPRMF attr.
	/// </summary>
	public class BPRMFAttr : BPRMF, IItemAttributeAwareRecommender
	{
		///
		public IBooleanMatrix ItemAttributes
		{
			get { return this.item_attributes; }
			set {
				this.item_attributes = value;
				this.NumItemAttributes = item_attributes.NumberOfColumns;
				this.MaxItemID = Math.Max(MaxItemID, item_attributes.NumberOfRows - 1);
			}
		}
		///
		protected IBooleanMatrix item_attributes;
		
		///
		public List<IBooleanMatrix> AdditionalItemAttributes
		{
			get { return this.additional_item_attributes; }
			set {
				this.additional_item_attributes = value;
			}
		}
		private List<IBooleanMatrix> additional_item_attributes;

		///
		public int NumItemAttributes { get; private set; }

		/// <summary>item factors (individual part)</summary>
		protected Matrix<float> q;

		/// <summary>user factors (individual part)</summary>
		protected internal Matrix<float> p;

		/// Item attribute weights
		protected Matrix<float> item_attribute_weight_by_user;

		/// <summary>
		/// Initializes a new instance of the <see cref="MyMediaLite.ItemRecommendation.BPRMFAttr"/> class.
		/// </summary>
		public BPRMFAttr()
		{
		}

		///
		protected override void InitModel()
		{
			base.InitModel ();

			user_factors = null;
			item_factors = null;
			p = new Matrix<float> (MaxUserID + 1, NumFactors);
			p.InitNormal (InitMean, InitStdDev);
			q = new Matrix<float> (MaxItemID + 1, NumFactors);
			q.InitNormal (InitMean, InitStdDev);

			Console.Write("Learning attributes...");
			BPRLinear learnAttr = new BPRLinear();
			learnAttr.Feedback = Feedback;
			learnAttr.ItemAttributes = item_attributes;
			learnAttr.NumIter = NumIter;//10;
			learnAttr.LearnRate = LearnRate;//0.05f;
			learnAttr.Regularization = 0.015f;//0.001f;
			learnAttr.Train();
			item_attribute_weight_by_user = learnAttr.ItemAttributeWeights;
			learnAttr = null;
			Console.WriteLine ("Done");
		}

		/// <summary>
		/// Sample a pair of items, given a user
		/// </summary>
		/// <param name='user_id'>
		/// the user ID
		/// </param>
		/// <param name='item_id'>
		/// the ID of the first item
		/// </param>
		/// <param name='other_item_id'>
		/// the ID of the second item
		/// </param>
		protected override void SampleItemPair(int user_id, out int item_id, out int other_item_id)
		{
			//SampleAnyItemPair(user_id, out item_id, out other_item_id);
			//base.SampleItemPair(user_id, out item_id, out other_item_id);
			//return;
			var user_items = Feedback.UserMatrix [user_id];
			item_id = user_items.ElementAt (random.Next (user_items.Count));

			if(item_id >= item_attributes.NumberOfRows) {
				do
					other_item_id = random.Next(MaxItemID + 1);
				while (user_items.Contains(other_item_id));
				return;
			}

			var attrList = item_attributes.GetEntriesByRow (item_id);
			float sum1 = 0;
			foreach (int g in attrList) {
				sum1 += item_attribute_weight_by_user[user_id, g];//weights [user_id, g];
			}
			if(attrList.Count > 0) sum1 /= attrList.Count;
			else sum1 = 0;

			while(true) {
				other_item_id = random.Next(MaxItemID + 1);
				if(!user_items.Contains(other_item_id)) {
					return;
				}

				if(other_item_id >= item_attributes.NumberOfRows)
					continue;

				attrList = item_attributes.GetEntriesByRow(other_item_id);
				float sum2 = 0;
				foreach(int g in attrList) {
					sum2 += item_attribute_weight_by_user[user_id, g];//weights[user_id, g];
				}
				if(attrList.Count > 0) sum2 /= attrList.Count;
				else sum2 = 0;
				
				if(Math.Abs(sum1-sum2) < 2.5)
					continue;
				
				if(sum1 < sum2) {
					int aux = item_id;
					item_id = other_item_id;
					other_item_id = aux;
				}
				return;				
			}
		}

		/// <summary>Update latent factors according to the stochastic gradient descent update rule</summary>
		/// <param name="user_id">the user ID</param>
		/// <param name="item_id">the ID of the first item</param>
		/// <param name="other_item_id">the ID of the second item</param>
		/// <param name="update_u">if true, update the user latent factors</param>
		/// <param name="update_i">if true, update the latent factors of the first item</param>
		/// <param name="update_j">if true, update the latent factors of the second item</param>
		protected override void UpdateFactors(int user_id, int item_id, int other_item_id, bool update_u, bool update_i, bool update_j)
		{

			double dotProductDiff = 0;
			 for (int c = 0; c < NumFactors; c++) 
			{
				dotProductDiff += p[user_id, c] * (q[item_id, c] - q[other_item_id, c]);
			}

			double x_uij = item_bias[item_id] - item_bias[other_item_id] + dotProductDiff;

			double one_over_one_plus_ex = 1 / (1 + Math.Exp(x_uij));

			// adjust bias terms
			if (update_i)
			{
				double update = one_over_one_plus_ex - BiasReg * item_bias[item_id];
				item_bias[item_id] += (float) (learn_rate * update);
			}

			if (update_j)
			{
				double update = -one_over_one_plus_ex - BiasReg * item_bias[other_item_id];
				item_bias[other_item_id] += (float) (learn_rate * update);
			}

			// adjust factors
			for (int f = 0; f < num_factors; f++)
			{
				float u_f = p[user_id, f];
				float i_f = q[item_id, f];
				float j_f = q[other_item_id, f];

				// if necessary, compute and apply updates
				if (update_u)
				{	
					double update = (i_f - j_f) * one_over_one_plus_ex - reg_u * u_f;
					p[user_id, f] = (float)(u_f + learn_rate * update);
				}
				if (update_i)
				{
					double update = u_f * one_over_one_plus_ex - reg_i * i_f;
					q[item_id, f] = (float)(i_f + learn_rate * update);
				}

				if(update_j) 
				{
					double update = -u_f * one_over_one_plus_ex - reg_j * j_f;
					q[other_item_id, f] = (float)(j_f + learn_rate * update);
				}
			}
		}

		///
		public override float Predict(int user_id, int item_id)
		{
			if (user_id > MaxUserID || item_id > MaxItemID)
				return float.MinValue;

			if (user_factors == null)
				PrecomputeUserFactors();

			if (item_factors == null)
				PrecomputeItemFactors();

			return item_bias[item_id] + DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);
		}

		/// <summary>Precompute all item factors</summary>
		protected void PrecomputeItemFactors()
		{
			if (item_factors == null)
				item_factors = new Matrix<float>(MaxItemID + 1, NumFactors);

			for (int item_id = 0; item_id <= MaxItemID; item_id++)
				PrecomputeItemFactors(item_id);
		}

		/// <summary>Precompute the factors for a given item</summary>
		/// <param name='item_id'>the ID of the item</param>
		protected void PrecomputeItemFactors(int item_id)
		{
			// assign
			for (int f = 0; f < NumFactors; f++) {
				item_factors [item_id, f] = (float)q[item_id, f];
			}
		}

		/// <summary>Precompute all user factors</summary>
		protected void PrecomputeUserFactors()
		{
			if (user_factors == null)
				user_factors = new Matrix<float>(MaxUserID + 1, NumFactors);

			for (int user_id = 0; user_id <= MaxUserID; user_id++)
				PrecomputeUserFactors(user_id);
		}

		/// <summary>Precompute the factors for a given user</summary>
		/// <param name='user_id'>the ID of the user</param>
		protected virtual void PrecomputeUserFactors(int user_id)
		{
			// assign
			for (int f = 0; f < NumFactors; f++) 
			{
				user_factors [user_id, f] = (float)p[user_id, f];
			}
		}

        public override void SaveModel(string file)
        {
            using (StreamWriter writer = Model.GetWriter(file, this.GetType(), "2.99"))
            {
                writer.WriteMatrix(user_factors);
                writer.WriteVector(item_bias);
                writer.WriteMatrix(item_factors);

                writer.WriteMatrix(item_attribute_weight_by_user);
                writer.WriteMatrix(q);
                writer.WriteMatrix(p);
                

            }
        }

        ///
        public override void LoadModel(string file)
        {
            using (StreamReader reader = Model.GetReader(file, this.GetType()))
            {
                var user_factors = (Matrix<float>)reader.ReadMatrix(new Matrix<float>(0, 0));
                var item_bias = reader.ReadVector();
                var item_factors = (Matrix<float>)reader.ReadMatrix(new Matrix<float>(0, 0));

                var item_attribute_weight_by_user = (Matrix<float>)reader.ReadMatrix(new Matrix<float>(0, 0));
                var q = (Matrix<float>)reader.ReadMatrix(new Matrix<float>(0, 0));
                var p = (Matrix<float>)reader.ReadMatrix(new Matrix<float>(0, 0));

                if (user_factors.NumberOfColumns != item_factors.NumberOfColumns)
                    throw new IOException(
                        string.Format(
                            "Number of user and item factors must match: {0} != {1}",
                            user_factors.NumberOfColumns, item_factors.NumberOfColumns));
                if (item_bias.Count != item_factors.dim1)
                    throw new IOException(
                        string.Format(
                            "Number of items must be the same for biases and factors: {0} != {1}",
                            item_bias.Count, item_factors.dim1));

                this.MaxUserID = user_factors.NumberOfRows - 1;
                this.MaxItemID = item_factors.NumberOfRows - 1;

                // assign new model
                if (this.num_factors != user_factors.NumberOfColumns)
                {
                    Console.Error.WriteLine("Set num_factors to {0}", user_factors.NumberOfColumns);
                    this.num_factors = user_factors.NumberOfColumns;
                }
                this.user_factors = user_factors;
                this.item_bias = (float[])item_bias;
                this.item_factors = item_factors;

                this.item_attribute_weight_by_user = item_attribute_weight_by_user;
                this.q = q;
                this.p = p;

            }
        }


		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"{0} num_factors={1} bias_reg={2} reg_u={3} reg_i={4} reg_j={5} num_iter={6} learn_rate={7} uniform_user_sampling={8} with_replacement={9} update_j={10} Decay={11}",
				this.GetType().Name, num_factors, BiasReg, reg_u, reg_i, reg_j, NumIter, learn_rate, UniformUserSampling, WithReplacement, UpdateJ, Decay);
		}
	}
}


