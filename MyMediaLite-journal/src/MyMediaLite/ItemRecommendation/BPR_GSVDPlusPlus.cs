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
	/// BP r_ GSVD plus plus.
	/// </summary>
	public class BPR_GSVDPlusPlus : BPRMF, IItemAttributeAwareRecommender
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

		/// <summary>item factors (part expressed via the items attributes)</summary>
		protected Matrix<float> x;

		/// <summary>item factors (individual part)</summary>
		protected Matrix<float> q;

		/// <summary>precomputed regularization terms for the x matrix</summary>
		protected float[] x_reg;

		/// <summary>precomputed regularization terms for the y matrix</summary>
		protected float[] y_reg;

		/// <summary>Regularization parameter for user factors</summary>
		public float RegX { get { return reg_x; } set { reg_x = value; } }
		/// <summary>Regularization parameter for user factors</summary>
		protected float reg_x = 20f;

		/// <summary>Regularization parameter for user factors</summary>
		public float RegY { get { return reg_y; } set { reg_y = value; } }
		/// <summary>Regularization parameter for user factors</summary>
		protected float reg_y = 2f;

		/// <summary>user factors (part expressed via the rated items)</summary>
		protected internal Matrix<float> y;

		/// <summary>user factors (individual part)</summary>
		protected internal Matrix<float> p;

		/// <summary>Regularization based on rating frequency</summary>
		/// <description>
		/// Regularization proportional to the inverse of the square root of the number of ratings associated with the user or item.
		/// As described in the paper by Menon and Elkan.
		/// </description>
		public bool FrequencyRegularization { get; set; }

		/// Item attribute weights
		protected Matrix<float> item_attribute_weight_by_user;

		/// <summary>
		/// Initializes a new instance of the <see cref="MyMediaLite.ItemRecommendation.BPR_GSVDPlusPlus"/> class.
		/// </summary>
		public BPR_GSVDPlusPlus()
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
			y = new Matrix<float> (MaxItemID + 1, NumFactors);
			y.InitNormal (InitMean, InitStdDev);
			x = new Matrix<float> (item_attributes.NumberOfColumns, NumFactors);
			x.InitNormal (InitMean, InitStdDev);
			q = new Matrix<float> (MaxItemID + 1, NumFactors);
			q.InitNormal (InitMean, InitStdDev);

			int num_attributes = item_attributes.NumberOfColumns;

			x_reg = new float[num_attributes];
			for (int attribute_id = 0; attribute_id < num_attributes; attribute_id++)
				x_reg [attribute_id] = FrequencyRegularization? (RegX / (float)(1 + Math.Exp(-0.005*item_attributes.NumEntriesByColumn (attribute_id)))) : RegX;

			y_reg = new float[MaxItemID + 1];
			for (int item_id = 0; item_id <= MaxItemID; item_id++) {
				var feedback_count_by_item = Feedback.ItemMatrix [item_id];
				if (feedback_count_by_item.Count > 0)
					y_reg [item_id] = FrequencyRegularization ? (float)(RegY / Math.Sqrt (feedback_count_by_item.Count)) : RegY;
				else
					y_reg [item_id] = 0;
			}

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

		protected void SampleAnyItemPair(int user_id, out int item_id, out int other_item_id) {
			var user_items = Feedback.UserMatrix [user_id];

			while(true) {
				item_id = random.Next(MaxItemID + 1);
				other_item_id = random.Next(MaxItemID + 1);

				if((user_items.Contains(item_id) &&  user_items.Contains (other_item_id)) ||
				  (!user_items.Contains(item_id) && !user_items.Contains (other_item_id))) {
					if(item_id >= item_attributes.NumberOfRows || other_item_id >= item_attributes.NumberOfRows) {
						continue;
					}

					var attrList = item_attributes.GetEntriesByRow (item_id);
					if(attrList.Count == 0) continue;
					float sum1 = 0;
					foreach (int g in attrList) {
						sum1 += item_attribute_weight_by_user[user_id, g];//weights [user_id, g];
					}
					sum1 /= attrList.Count;

					attrList = item_attributes.GetEntriesByRow (other_item_id);
					if(attrList.Count == 0) continue;
					float sum2 = 0;
					foreach (int g in attrList) {
						sum2 += item_attribute_weight_by_user[user_id, g];//weights [user_id, g];
					}
					sum2 /= attrList.Count;

					//Console.WriteLine (Math.Abs(sum1-sum2));
					if(Math.Abs(sum1-sum2) < 3.5)
						continue;

					if(sum1 < sum2) {
						int aux = other_item_id;
						other_item_id = item_id;
						item_id = aux;
					}
					return;
				}
				else {
					if(!user_items.Contains(item_id)) {
						int aux = other_item_id;
						other_item_id = item_id;
						item_id = aux;
					}
					return;
				}
			}
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
			var items_rated_by_user = Feedback.UserMatrix [user_id];
			var p_plus_y_sum_vector = y.SumOfRows (items_rated_by_user);
			double norm_denominator = (items_rated_by_user.Count > 0) ? Math.Sqrt (items_rated_by_user.Count) : 1;
			for (int f = 0; f < p_plus_y_sum_vector.Count; f++) {
				p_plus_y_sum_vector [f] = (float)(p_plus_y_sum_vector [f] / norm_denominator + p [user_id, f]);
			}

			var q_plus_x_sum_vector = q.GetRow (item_id);

			if (item_id < item_attributes.NumberOfRows) {
				IList<int> attribute_list = item_attributes.GetEntriesByRow (item_id);
				double second_norm_denominator = attribute_list.Count;
				if(second_norm_denominator > 0) {
					var x_sum_vector = x.SumOfRows (attribute_list);
					for (int f = 0; f < x_sum_vector.Count; f++)
						q_plus_x_sum_vector [f] += (float)(x_sum_vector [f] / second_norm_denominator);
				}
			}

			var other_q_plus_x_sum_vector = q.GetRow (other_item_id);

			if (other_item_id < item_attributes.NumberOfRows) {
				IList<int> attribute_list = item_attributes.GetEntriesByRow (other_item_id);
				double second_norm_denominator = attribute_list.Count;
				if(second_norm_denominator > 0) {
					var x_sum_vector = x.SumOfRows (attribute_list);
					for (int f = 0; f < x_sum_vector.Count; f++)
						other_q_plus_x_sum_vector [f] += (float)(x_sum_vector [f] / second_norm_denominator);
				}
			}

			double dotProductDiff = 0;
			for (int c = 0; c < p_plus_y_sum_vector.Count; c++) 
			{
				dotProductDiff += p_plus_y_sum_vector [c] * (q_plus_x_sum_vector [c] - other_q_plus_x_sum_vector [c]);
				//dotProductDiff += p[user_id, c] * (q_plus_x_sum_vector [c] - other_q_plus_x_sum_vector [c]);
				//dotProductDiff += p[user_id, c] * (q[item_id, c] - q[other_item_id, c]);
				//Console.Write (dotProductDiff + " ");
			}
			//Console.WriteLine ();

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
			double normalized_error = one_over_one_plus_ex / norm_denominator;
			for (int f = 0; f < num_factors; f++)
			{
				float u_f = p_plus_y_sum_vector[f];
				//float u_f = p[user_id, f];
				float i_f = q_plus_x_sum_vector[f];
				//float i_f = q[item_id, f];
				float j_f = other_q_plus_x_sum_vector[f];
				//float j_f = q[other_item_id, f];
				float p_f = p[user_id, f];
				float q_f = q[item_id, f];
				float other_q_f = q[other_item_id, f];
				float[] x_f = new float[NumItemAttributes];
				for(int g = 0; g < NumItemAttributes; g++) 
				{
					x_f[g] = x[g, f];
				}
				float[] y_f = new float[MaxItemID + 1];
				for (int g = 0; g <= MaxItemID; g++) 
				{
					y_f[g] = y[g, f];
				}

				// if necessary, compute and apply updates
				if (update_u)
				{	
					double update = (i_f - j_f) * one_over_one_plus_ex - reg_u * p_f;
					p[user_id, f] = (float)(p_f + learn_rate * update);
				}
				if (update_i)
				{
					double update = u_f * one_over_one_plus_ex - reg_i * q_f;
					q[item_id, f] = (float)(q_f + learn_rate * update);
					
					foreach (int other_y in items_rated_by_user)
					{
						double delta = (i_f - j_f) * normalized_error - y_reg[other_y] * y_f[other_y];
						y[other_y, f] = (float)(y_f[other_y] + learn_rate * delta);
					}

					// adjust attributes
					if (item_id < item_attributes.NumberOfRows)
					{
						IList<int> attribute_list = item_attributes.GetEntriesByRow(item_id);
						double second_norm_denominator = attribute_list.Count;
						double second_norm_error = one_over_one_plus_ex / second_norm_denominator;
						foreach (int attribute_id in attribute_list)
						{
							double delta = u_f * second_norm_error - x_reg[attribute_id] * x_f[attribute_id];
							x[attribute_id, f] = (float)(x_f[attribute_id] + learn_rate * delta);
						}
					}
				}

				if(update_j) 
				{
					double update = -u_f * one_over_one_plus_ex - reg_j * other_q_f;
					q[other_item_id, f] = (float)(other_q_f + learn_rate * update);

					// adjust attributes
					if (other_item_id < item_attributes.NumberOfRows)
					{
						IList<int> attribute_list = item_attributes.GetEntriesByRow(other_item_id);
						double second_norm_denominator = attribute_list.Count;
						double second_norm_error = one_over_one_plus_ex / second_norm_denominator;
						foreach (int attribute_id in attribute_list)
						{
							double delta = -u_f * second_norm_error - x_reg[attribute_id] * x_f[attribute_id];
							x[attribute_id, f] = (float)(x_f[attribute_id] + learn_rate * delta);
						}
					}
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
			
			float predicted = item_bias[item_id] + DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);
			return predicted;
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
			// compute
			var factors = q.GetRow(item_id);
			if (item_id < item_attributes.NumberOfRows)
			{
				IList<int> attribute_list = item_attributes.GetEntriesByRow(item_id);
				double second_norm_denominator = attribute_list.Count;
				
				// o if abaixo, se retirado com a base ML, retorna melhores resultados. Com outras bases deve-se mante-lo.
				if(second_norm_denominator > 0) {
					var x_sum_vector = x.SumOfRows(attribute_list);
					for (int f = 0; f < x_sum_vector.Count; f++)
						factors[f] += (float) (x_sum_vector[f] / second_norm_denominator);
				}
			}

			// assign
			for (int f = 0; f < NumFactors; f++) {
				item_factors [item_id, f] = (float)factors[f];
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
			var items_rated_by_user = Feedback.UserMatrix [user_id];

			if (items_rated_by_user.Count == 0)
				return;

			// compute
			var factors = y.SumOfRows(items_rated_by_user);
			double norm_denominator = Math.Sqrt(items_rated_by_user.Count);
			for (int f = 0; f < factors.Count; f++)
				factors[f] = (float) (factors[f] / norm_denominator + p[user_id, f]);

			// assign
			for (int f = 0; f < NumFactors; f++) 
			{
				user_factors [user_id, f] = (float)factors [f];
				//user_factors [user_id, f] = (float)p[user_id, f];
			}
		}

		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"{0} num_factors={1} bias_reg={2} reg_u={3} reg_i={4} reg_j={5} reg_x={6} reg_y={7} num_iter={8} learn_rate={9} uniform_user_sampling={10} with_replacement={11} update_j={12} frequency_regularization={13} Decay={14}",
				this.GetType().Name, num_factors, BiasReg, reg_u, reg_i, reg_j, reg_x, reg_y, NumIter, learn_rate, UniformUserSampling, WithReplacement, UpdateJ, FrequencyRegularization, Decay);
		}
	}
}


